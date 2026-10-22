"""Gap-filling completion for zarr-based Earth System Data Cube datasets.

Public API:
    fill_dataset(predict_fn, path, var, aux_vars, block_size, patch_overlap, norm_range, batch_size)

predict_fn is a callable with signature:
    batch_np: np.ndarray  # (B, T, H, W, C) channels-last float32
    -> np.ndarray          # (B, C_out, H, W) channels-first float32

The function is model-agnostic: no torch or TF imports here.
"""

import itertools
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import tqdm
import xarray as xr
import zarr

from patchfill.preprocessing import calculate_total_chunks, get_chunk_sizes, reverse_normalization


def _iter_blocks_with_padding(
    ds: xr.Dataset,
    block_size: Tuple[int, int, int],
    patch_overlap: int,
) -> Iterator[Tuple[Dict[str, np.ndarray], Dict[str, slice]]]:
    """Yield (var_blocks, write_idx) for every zarr chunk in ds.

    var_blocks maps each variable name to an ndarray of shape
        (chunk_T + 2*pad_t, chunk_H + 2*pad_s, chunk_W + 2*pad_s),
    NaN-padded at dataset boundaries so every position in the write region
    has full temporal and spatial context.

    write_idx is {'time': slice, 'lat': slice, 'lon': slice} pointing to
    the unpadded chunk region in the output zarr.
    """
    chunk_sizes = get_chunk_sizes(ds)
    dim_names = [cs[0] for cs in chunk_sizes]
    dim_chunk = {cs[0]: cs[1] for cs in chunk_sizes}
    dim_size = {d: ds.sizes[d] for d in dim_names}

    pad_t = (block_size[0] - 1) // 2
    pad_s = patch_overlap

    dim_ranges = [range(0, dim_size[d], dim_chunk[d]) for d in dim_names]

    for offsets in itertools.product(*dim_ranges):
        t0, h0, w0 = offsets

        # Canonical (unpadded) chunk boundaries
        t1 = min(t0 + dim_chunk['time'], dim_size['time'])
        h1 = min(h0 + dim_chunk['lat'],  dim_size['lat'])
        w1 = min(w0 + dim_chunk['lon'],  dim_size['lon'])

        write_idx = {
            'time': slice(t0, t1),
            'lat':  slice(h0, h1),
            'lon':  slice(w0, w1),
        }

        # Padded read boundaries (clamped to dataset)
        rt0 = max(0, t0 - pad_t);  rt1 = min(dim_size['time'], t1 + pad_t)
        rh0 = max(0, h0 - pad_s);  rh1 = min(dim_size['lat'],  h1 + pad_s)
        rw0 = max(0, w0 - pad_s);  rw1 = min(dim_size['lon'],  w1 + pad_s)

        # NaN rows to prepend/append: deficit between desired and available padding
        pre_t  = pad_t - (t0 - rt0);  post_t = pad_t - (rt1 - t1)
        pre_h  = pad_s - (h0 - rh0);  post_h = pad_s - (rh1 - h1)
        pre_w  = pad_s - (w0 - rw0);  post_w = pad_s - (rw1 - w1)

        var_blocks: Dict[str, np.ndarray] = {}

        for var_name, var in ds.data_vars.items():
            read_idx = {
                'time': slice(rt0, rt1),
                'lat':  slice(rh0, rh1),
                'lon':  slice(rw0, rw1),
            }
            arr = var.isel(
                {d: read_idx[d] for d in read_idx if d in var.sizes}
            ).values.astype(np.float32)

            def _pad(a, axis, pre, post):
                shape = list(a.shape)
                if pre > 0:
                    shape[axis] = pre
                    a = np.concatenate([np.full(shape, np.nan, dtype=np.float32), a], axis=axis)
                if post > 0:
                    shape = list(a.shape)
                    shape[axis] = post
                    a = np.concatenate([a, np.full(shape, np.nan, dtype=np.float32)], axis=axis)
                return a

            arr = _pad(arr, 0, pre_t, post_t)
            arr = _pad(arr, 1, pre_h, post_h)
            arr = _pad(arr, 2, pre_w, post_w)

            var_blocks[var_name] = arr

        yield var_blocks, write_idx


def _create_output_var(
    path: str,
    var: str,
    output_name: str = 'no_gaps.zarr',
    time_slice: Optional[slice] = None,
    preprocessed_name: str = 'preprocessed.zarr',
) -> None:
    """Copy preprocessed_name to output_name, adding an empty <var>_no_gaps variable.

    If time_slice is provided, only the matching time window is written.
    """
    ds = xr.open_zarr(path + preprocessed_name)
    if time_slice is not None:
        ds = ds.sel(time=time_slice)
    ds[var + '_no_gaps'] = ds[var].copy(deep=True)
    for v in ds.variables:
        ds[v].encoding = {}
    ds.to_zarr(path + output_name, consolidated=True, safe_chunks=False, mode='w')


def _write_predictions(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    batch: List[np.ndarray],
    indices: List[Dict[str, int]],
    zarr_chunk: np.ndarray,
    block_size: Tuple[int, int, int],
    patch_overlap: int,
) -> np.ndarray:
    """Run predict_fn on the batch, merge with originals, write to zarr_chunk."""
    batch_np = np.stack(batch, axis=0)  # (B, T, H+2*ov, W+2*ov, C)
    pred = predict_fn(batch_np)          # (B, C_out, H+2*ov, W+2*ov)

    mid = block_size[0] // 2
    s = slice(patch_overlap, -patch_overlap if patch_overlap > 0 else None)

    orig_mid = batch_np[:, mid, s, s, 0]   # (B, H, W) — LST channel of middle frame
    pred_center = pred[:, 0, s, s]         # (B, H, W) — center of prediction

    merged = pred_center.copy()
    valid = ~np.isnan(orig_mid)
    merged[valid] = orig_mid[valid]         # keep original where not NaN

    for pos, pred_2d in zip(indices, merged):
        h = min(block_size[1], zarr_chunk.shape[1] - pos['lat'])
        w = min(block_size[2], zarr_chunk.shape[2] - pos['lon'])
        zarr_chunk[
            pos['time'],
            pos['lat']:pos['lat'] + h,
            pos['lon']:pos['lon'] + w,
        ] = pred_2d[:h, :w]

    return zarr_chunk


def _fill_chunk(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    zarr_store: zarr.Group,
    var: str,
    var_data: np.ndarray,
    land_mask: np.ndarray,
    write_idx: Dict[str, slice],
    block_size: Tuple[int, int, int],
    patch_overlap: int,
    batch_size: int,
) -> None:
    """Fill all patchable positions within a single zarr chunk."""
    out_key = var + '_no_gaps'
    t0 = write_idx['time'].start;  t1 = write_idx['time'].stop
    h0 = write_idx['lat'].start;   h1 = write_idx['lat'].stop
    w0 = write_idx['lon'].start;   w1 = write_idx['lon'].stop

    zarr_chunk = zarr_store[out_key][t0:t1, h0:h1, w0:w1]

    T_padded, H_padded, W_padded, _ = var_data.shape
    mid = block_size[0] // 2
    bH = block_size[1] + 2 * patch_overlap
    bW = block_size[2] + 2 * patch_overlap
    s = slice(patch_overlap, -patch_overlap if patch_overlap > 0 else None)

    batch: List[np.ndarray] = []
    indices: List[Dict[str, int]] = []

    for t in range(0, T_padded - (block_size[0] - 1)):
        for lat in range(0, H_padded - 2 * patch_overlap, block_size[1]):
            for lon in range(0, W_padded - 2 * patch_overlap, block_size[2]):

                block = var_data[t:t + block_size[0], lat:lat + bH, lon:lon + bW]

                if block.shape[0] < block_size[0] or block.shape[1] < bH or block.shape[2] < bW:
                    continue  # undersized edge block (safety guard, rarely triggered)

                # Land mask for the unpadded middle-frame region
                lm = land_mask[t + mid, lat + patch_overlap:lat + patch_overlap + block_size[1],
                               lon + patch_overlap:lon + patch_overlap + block_size[2]]
                if not np.any(lm):
                    continue  # all ocean

                mid_vals = block[mid, s, s, 0]
                if not np.any(np.isnan(mid_vals[lm])):
                    continue  # no gaps on land in middle frame

                if np.all(np.isnan(block)):
                    continue  # entirely NaN — model has no signal

                batch.append(block)
                indices.append({'time': t, 'lat': lat, 'lon': lon})

                if len(batch) >= batch_size:
                    zarr_chunk = _write_predictions(
                        predict_fn, batch, indices, zarr_chunk, block_size, patch_overlap
                    )
                    batch = []
                    indices = []

    if batch:
        zarr_chunk = _write_predictions(
            predict_fn, batch, indices, zarr_chunk, block_size, patch_overlap
        )

    zarr_store[out_key][t0:t1, h0:h1, w0:w1] = zarr_chunk


def _reverse_normalize_inplace(
    zarr_store: zarr.Group,
    var: str,
    norm_range: Tuple[float, float],
    batch_size: int = 10,
) -> None:
    print(f'Denormalizing {var} ...')
    n = zarr_store[var].shape[0]
    for start in tqdm.tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        zarr_store[var][start:end] = reverse_normalization(zarr_store[var][start:end], norm_range)


def _apply_land_mask_inplace(
    zarr_store: zarr.Group,
    var: str,
    batch_size: int = 10,
) -> None:
    print('Applying land mask ...')
    n = zarr_store[var].shape[0]
    for start in tqdm.tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        zarr_store[var][start:end] = np.where(
            zarr_store['land_mask'][start:end],
            zarr_store[var][start:end],
            np.nan,
        )


def fill_dataset(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    path: str,
    var: str,
    aux_vars: List[Dict],
    block_size: Tuple[int, int, int],
    patch_overlap: int,
    norm_range: Optional[Tuple[float, float]],
    batch_size: int = 32,
    output_name: str = 'no_gaps.zarr',
    time_slice: Optional[slice] = None,
    preprocessed_name: str = 'preprocessed.zarr',
) -> None:
    """Fill gaps in the preprocessed zarr and write results to output_name.

    Steps:
      1. Create output_name with a <var>_no_gaps variable (copy of <var>).
         If time_slice is provided, only that time window is written.
      2. Iterate chunks; for each, collect gappy patches, call predict_fn,
         merge predictions with original data, write back.
      3. Optionally reverse-normalize both <var> and <var>_no_gaps.
      4. Apply land mask to <var>_no_gaps (NaN over ocean/polar).
    """
    print('Creating output zarr ...')
    _create_output_var(path, var, output_name, time_slice, preprocessed_name)

    ds = xr.open_zarr(path + preprocessed_name)
    if time_slice is not None:
        ds = ds.sel(time=time_slice)
    zarr_store = zarr.open(path + output_name, mode='r+')

    total = calculate_total_chunks(ds)
    with tqdm.tqdm(total=total, desc='Filling chunks') as pbar:
        for var_blocks, write_idx in _iter_blocks_with_padding(ds, block_size, patch_overlap):
            pbar.update(1)

            var_data = np.expand_dims(var_blocks[var], axis=-1)
            for aux in aux_vars:
                aux_data = np.expand_dims(var_blocks[aux['var']], axis=-1)
                var_data = np.concatenate([var_data, aux_data], axis=-1)

            land_mask = var_blocks['land_mask'].astype(bool)

            _fill_chunk(
                predict_fn, zarr_store, var, var_data, land_mask,
                write_idx, block_size, patch_overlap, batch_size,
            )

    if norm_range is not None:
        _reverse_normalize_inplace(zarr_store, var, norm_range)
        _reverse_normalize_inplace(zarr_store, var + '_no_gaps', norm_range)

    _apply_land_mask_inplace(zarr_store, var + '_no_gaps')
