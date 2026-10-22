import itertools
import time
import dask
import dask.array as da
import numpy as np
from global_land_mask import globe
import tqdm
from typing import Tuple, Iterator, Dict, List
import xarray as xr
import zarr
import shutil
import os
import fsspec
from fsspec.config import conf as _fsspec_conf

# Limit concurrent HTTP chunk fetches to avoid 429 rate limits from the ESDC server.
_fsspec_conf["nofiles_gather_batch_size"] = 1
_fsspec_conf["gather_batch_size"] = 1
if hasattr(zarr, "config"):  # zarr v3+
    zarr.config.set({"async.concurrency": 1})


def iter_data_var_blocks(ds: xr.Dataset, block_size: List[Tuple[str, int]] = None) \
        -> Iterator[Dict[str, np.ndarray]]:
    """Yield data blocks for all variables in the dataset."""
    block_size = get_chunk_sizes(ds) if block_size is None else block_size
    dim_ranges = []
    for dim_name, chunk_size in block_size:
        dim_size = ds.sizes[dim_name]
        dim_ranges.append(range(0, dim_size, chunk_size))
    for offsets in itertools.product(*dim_ranges):
        dim_slices = {bs[0]: slice(offset, offset + bs[1])
                      for bs, offset in zip(block_size, offsets)}
        var_blocks = {}
        for var_name, var in ds.data_vars.items():
            indexers = {dim_name: dim_slice
                        for dim_name, dim_slice in dim_slices.items()
                        if dim_name in var.sizes}
            var_blocks[var_name] = var.isel(indexers).values
        yield var_blocks


def get_chunk_sizes(ds: xr.Dataset) -> List[Tuple[str, int]]:
    """Return the maximum chunk size per dimension across all data variables."""
    chunk_sizes = {}
    for var in ds.data_vars.values():
        if var.chunks:
            chunks = tuple(max(*c) if len(c) > 1 else c[0] for c in var.chunks)
            for dim_name, chunk_size in zip(var.sizes, chunks):
                chunk_sizes[dim_name] = max(chunk_size, chunk_sizes.get(dim_name, 0))
    return [(str(k), v) for k, v in chunk_sizes.items()]


def calculate_total_chunks(ds: xr.Dataset, block_size: List[Tuple[str, int]] = None) -> int:
    """Return the total number of chunks for the dataset.

    If block_size is provided, those sizes override the dataset's own chunk sizes.
    """
    effective_sizes = dict(get_chunk_sizes(ds))
    if block_size is not None:
        for dim, size in block_size:
            if size is not None:
                effective_sizes[dim] = size
    return int(np.prod([
        len(range(0, ds.sizes[dim], size))
        for dim, size in effective_sizes.items()
    ]))


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Drop inherited zarr v2 codec encodings so zarr v3 doesn't choke on them."""
    for v in ds.variables:
        ds[v].encoding = {}
    return ds


def _to_zarr_with_retry(
    ds: xr.Dataset,
    path: str,
    *,
    retries: int = 8,
    base_delay: float = 30.0,
    inter_slice_sleep: float = 2.0,
) -> None:
    """Write ds to a local zarr in temporal chunks, retrying on 429 errors.

    Initialises the store structure first (writes coordinates, no data fetched),
    then writes each temporal chunk via region writes so retries are idempotent —
    a 429 mid-way only costs that one chunk, not the whole dataset.

    inter_slice_sleep adds a proactive pause between slices: each zarr chunk is a
    separate HTTP file so requests cannot be batched, but spacing bursts keeps the
    sustained request rate below the server's rate limit.
    """
    time_chunk = ds.chunks.get('time', [100])[0]

    with dask.config.set(scheduler="synchronous"):
        ds.to_zarr(path, compute=False, consolidated=False, safe_chunks=False, mode='w')
    # Consolidate so region writes don't fall back to slow non-consolidated reads.
    zarr.consolidate_metadata(path)

    # Coords without a time dimension (lat, lon) were written during init and must
    # be dropped from region-write slices — xarray requires all variables in a
    # region write to share at least one dimension with the region.
    non_time_coords = [v for v in ds.coords if 'time' not in ds[v].dims]

    n_times = ds.sizes['time']
    for t_start in tqdm.trange(0, n_times, time_chunk, desc='Writing zarr slices', unit='chunk'):
        t_end = min(t_start + time_chunk, n_times)
        slice_ds = ds.isel(time=slice(t_start, t_end)).drop_vars(non_time_coords)
        for attempt in range(retries):
            try:
                with dask.config.set(scheduler="synchronous"):
                    slice_ds.to_zarr(path, region={'time': slice(t_start, t_end)}, safe_chunks=False)
                break
            except Exception as exc:
                if '429' in str(exc) and attempt < retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"\n  429 Too Many Requests at time steps [{t_start}:{t_end}], "
                          f"retrying in {delay:.0f}s …")
                    time.sleep(delay)
                else:
                    raise
        if t_end < n_times:
            time.sleep(inter_slice_sleep)

    zarr.consolidate_metadata(path)


def reverse_normalization(arr, norm_range):
    """Invert min-max normalization."""
    return arr * (norm_range[1] - norm_range[0]) + norm_range[0]


def clear_directory(dir_path):
    """Remove a directory and all its contents if it exists."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def get_esdc_data(var, aux_vars, time_span):
    """Fetch the specified variables from the remote ESDC zarr store."""
    url = "https://data.rsc4earth.de/download/EarthSystemDataCube/v2.1.1/esdc-8d-0.083deg-184x270x270-2.1.1.zarr"
    fs = fsspec.filesystem('https')
    ds = xr.open_zarr(fs.get_mapper(url), consolidated=True)
    ds = ds[[var] + [v['var'] for v in aux_vars]]
    ds = ds.sel(time=time_span)
    return ds


def preprocess(var, output_path, norm_range, block_size, patch_overlap, aux_vars, time_span, valid_range=None, preprocessed_name='preprocessed.zarr'):
    """Fetch, rechunk, clean, normalise, and apply land mask — single zarr write."""
    print('Fetching data ...')
    ds = get_esdc_data(var, aux_vars, time_span)

    # Chunk sizes are set to multiples of the block size so each chunk contains many
    # blocks, keeping I/O overhead low while staying within memory limits.
    temporal_chunk_factor = 20
    spatial_chunk_factor = 4
    spatial_size = block_size[1] + 2 * patch_overlap
    chunk_size = (
        block_size[0] * temporal_chunk_factor,
        spatial_size * spatial_chunk_factor,
        spatial_size * spatial_chunk_factor,
    )

    print('Applying transformations (lazy) ...')

    for var_name in [var] + [v['var'] for v in aux_vars]:
        ds[var_name] = (ds[var_name].dims,
                        da.rechunk(ds[var_name].data, chunks=chunk_size, block_size_limit=1e7))

    if valid_range is not None:
        ds[var] = ds[var].where((ds[var] >= valid_range[0]) & (ds[var] <= valid_range[1]))
    for aux in aux_vars:
        vr = aux.get('valid_range')
        if vr is not None:
            ds[aux['var']] = ds[aux['var']].where(
                (ds[aux['var']] >= vr[0]) & (ds[aux['var']] <= vr[1])
            )

    if norm_range:
        ds[var] = (ds[var] - norm_range[0]) / (norm_range[1] - norm_range[0])
    for aux in aux_vars:
        av_norm = aux.get('valid_range') or aux['computed_range']
        ds[aux['var']] = (ds[aux['var']] - av_norm[0]) / (av_norm[1] - av_norm[0])

    lon_grid, lat_grid = np.meshgrid(ds.lon.values, ds.lat.values)
    lm0 = da.from_array(globe.is_land(lat_grid, lon_grid)).rechunk((chunk_size[1], chunk_size[2]))
    lm = da.stack([lm0] * ds.sizes['time'], axis=0).rechunk(chunk_size)
    ds = ds.assign(land_mask=(['time', 'lat', 'lon'], lm))
    min_latitude = -83.87
    ds['land_mask'] = ds['land_mask'].where(ds['lat'] >= min_latitude, other=False)
    ds[var] = ds[var].where(ds['land_mask'])

    clear_directory(output_path + preprocessed_name)
    _clear_encoding(ds)
    print('Writing preprocessed data to zarr ...')
    _to_zarr_with_retry(ds, output_path + preprocessed_name)
