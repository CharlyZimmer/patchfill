import xarray as xr
import dask.array as da
import numpy as np
import random
import tqdm
import h5py
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Iterator, Dict
import zarr
import itertools
import regionmask

from patchfill.preprocessing import calculate_total_chunks, get_chunk_sizes


@dataclass
class BatchData:
    """Container for a batch of patch samples accumulated before writing to HDF5."""
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)
    time: list = field(default_factory=list)
    lat: list = field(default_factory=list)
    lon: list = field(default_factory=list)
    donor_masks: list = field(default_factory=list)

    def __len__(self):
        return len(self.X)

    def append(self, block_X, block_y, block_time, block_lat, block_lon, donor_mask=None):
        self.X.append(block_X)
        self.y.append(block_y)
        self.time.append(block_time)
        self.lat.append(block_lat)
        self.lon.append(block_lon)
        if donor_mask is not None:
            self.donor_masks.append(donor_mask)

    def clear(self):
        self.X.clear()
        self.y.clear()
        self.time.clear()
        self.lat.clear()
        self.lon.clear()
        self.donor_masks.clear()


def assign_block_split(x: xr.Dataset, block_size: Sequence[Tuple[str, int]] = None, split: float = 0.8,
                        seed: int = None):
    """Assign each spatial block to train or test with probability split / 1-split.

    The assignment is deterministic per block (via Cantor pairing), so reruns produce
    the same split. When seed is None, the split is derived purely from block position
    (legacy behavior). When seed is set, it is mixed into the hash so that two datasets
    built over the same grid/block_size with the same seed get an identical split.

    Usage:
        xds = xr.open_zarr("***.zarr")
        cube_with_split = assign_block_split(xds, block_size=[("time", 10), ("lat", 20), ("lon", 20)], split=0.5)
    """
    print('Assigning Block Split ...')

    def _assign_split(x, block_id=None):
        if block_id is not None:
            if seed is None:
                random.seed(cantor_tuple(block_id))
            else:
                random.seed(cantor_tuple(list(block_id) + [seed]))
        return x + (random.random() < split)

    def block_rand(x):
        block_ind_array = da.zeros(
            list(x.dims.values()), chunks=[v for _, v in block_size]
        )
        mapped = block_ind_array.map_blocks(_assign_split)
        mapped = mapped.astype(bool)
        mapped = da.invert(mapped)
        return ("time", "lat", "lon"), mapped

    return x.assign({"split": block_rand})


def assign_sloo_split(x: xr.Dataset, block_size, split_ranges):
    """Assign a spatial leave-one-out split based on AR6 climate regions.

    Args:
        split_ranges: (start_region, end_region) index pair selecting which AR6 regions go to test.
    """
    print('Assigning SLOO Split ...')

    lon_grid, lat_grid = np.meshgrid(x.lon, x.lat)
    mask = regionmask.defined_regions.ar6.all.mask(lon_grid, lat_grid)
    mask = mask.isin(list(range(split_ranges[0], split_ranges[1])))
    mask_stacked = da.stack([mask for _ in range(x.dims['time'])], axis=0)

    return x.assign({"split": (
        ("time", "lat", "lon"),
        mask_stacked.rechunk(chunks=[v for _, v in get_chunk_sizes(x)])
    )})


def _fetch_donor_mask_series(all_times_data, lat_start, lat_stop, lon_start, lon_stop,
                              current_t_start, block_size_t,
                              min_gap_frac=0.05, max_gap_frac=0.85):
    """Select a donor gap mask time series from a non-overlapping time window.

    The window must satisfy:
    - Combined gap fraction across all T frames in [min_gap_frac, max_gap_frac]
    - Middle frame gap fraction >= min_gap_frac (ensures a non-trivial reconstruction target)

    Returns a (T, H_block, W_block) boolean array or None if no valid window exists.
    """
    n_times = all_times_data.shape[0]
    middle_t = block_size_t // 2
    available_starts = [
        k for k in range(n_times - block_size_t + 1)
        if k + block_size_t <= current_t_start or k >= current_t_start + block_size_t
    ]
    if not available_starts:
        return None

    # Compute per-frame NaN fractions once for this spatial patch across all time steps.
    spatial = all_times_data[:, lat_start:lat_stop, lon_start:lon_stop]
    nan_mask = np.isnan(spatial)  # (n_times, H, W)
    frame_fracs = nan_mask.mean(axis=(1, 2))  # (n_times,)

    valid_starts = [
        k for k in available_starts
        if min_gap_frac <= frame_fracs[k:k + block_size_t].mean() <= max_gap_frac
        and frame_fracs[k + middle_t] >= min_gap_frac
    ]
    if not valid_starts:
        return None

    k = random.choice(valid_starts)
    return nan_mask[k:k + block_size_t]


def generate_synthetic_gaps(batch_X, donor_masks):
    """Inject synthetic gaps into all time steps of each sample using donor mask series.

    Each donor mask is a (T, H, W) boolean array derived from a non-overlapping time
    window at the same spatial location, making gaps temporally coherent across frames.
    """
    for sample, donor_mask_series in zip(batch_X, donor_masks):
        for t in range(sample.shape[0]):
            sample[t, ..., 0][donor_mask_series[t]] = np.nan
    return batch_X


def append_h5(X, y, time, lat, lon, batch: BatchData, num_samples, shape_X, shape_y, patch_overlap, syn_gaps=False, record_metainfo=True):
    """Append a batch of samples to an open HDF5 split (train or test).

    Writes in random-insertion order to shuffle the dataset on the fly. When syn_gaps
    is True, synthetic cloud gaps are injected into the middle input time step and the
    corresponding valid area in the target is masked to exclude it from the loss.
    """
    batch_X = np.array(batch.X)
    batch_y = np.array(batch.y)
    if record_metainfo:
        batch_time = np.array(batch.time)
        batch_lat = np.array(batch.lat)
        batch_lon = np.array(batch.lon)

    if syn_gaps:
        batch_X = generate_synthetic_gaps(batch_X, batch.donor_masks)
        # Mask the valid area in the target so the loss ignores regions without gaps.
        middle_valid = ~np.isnan(batch_X[:, int((shape_X[0] - 1) * 0.5), ..., 0])
        mask = np.expand_dims(np.expand_dims(middle_valid, axis=1), axis=4)
        batch_y[mask] = np.nan

    non_empty = ~np.all(np.isnan(batch_y), axis=(1, 2, 3, 4))
    batch_X = batch_X[non_empty]
    batch_y = batch_y[non_empty]
    if record_metainfo:
        batch_time = batch_time[non_empty]
        batch_lat = batch_lat[non_empty]
        batch_lon = batch_lon[non_empty]

    batch_len = len(batch_X)
    new_size = num_samples + batch_len
    X.resize((new_size,) + shape_X)
    y.resize((new_size,) + shape_y)
    if record_metainfo:
        time.resize((new_size, 2))
        lat.resize((new_size,))
        lon.resize((new_size,))

    if num_samples <= batch_len:
        X[num_samples:new_size] = batch_X
        y[num_samples:new_size] = batch_y
        if record_metainfo:
            time[num_samples:new_size] = batch_time
            lat[num_samples:new_size] = batch_lat
            lon[num_samples:new_size] = batch_lon
    else:
        # Insert at a random position to shuffle the dataset on the fly.
        position = random.randint(0, num_samples - batch_len)
        X[num_samples:new_size] = X[position:position + batch_len]
        X[position:position + batch_len] = batch_X
        y[num_samples:new_size] = y[position:position + batch_len]
        y[position:position + batch_len] = batch_y
        if record_metainfo:
            time[num_samples:new_size] = time[position:position + batch_len]
            time[position:position + batch_len] = batch_time
            lat[num_samples:new_size] = lat[position:position + batch_len]
            lat[position:position + batch_len] = batch_lat
            lon[num_samples:new_size] = lon[position:position + batch_len]
            lon[position:position + batch_len] = batch_lon

    return new_size


def _write_batch(X_ds, y_ds, time_ds, lat_ds, lon_ds, batch: BatchData, num_samples, shape_X, shape_y, patch_overlap, syn_gaps):
    """Flush a batch to HDF5, reset the batch container, and return the updated sample count."""
    updated = append_h5(X_ds, y_ds, time_ds, lat_ds, lon_ds, batch, num_samples, shape_X, shape_y, patch_overlap, syn_gaps=syn_gaps)
    batch.clear()
    return updated

def cantor_pairing(x: int, y: int):
    """Bijective mapping of a pair (x, y) to a natural number."""
    return int((x + y) * (x + y + 1) / 2 + y)


def cantor_tuple(index_list: list):
    """Generalise Cantor pairing to tuples for deterministic per-block random seeds."""
    t = index_list[0]
    for x in index_list[1:]:
        t = cantor_pairing(t, x)
    return t


def iter_data_var_blocks_with_indexers(ds: xr.Dataset, block_size, strides) \
        -> Iterator[Dict[str, np.ndarray]]:
    """Yield data blocks with their coordinates and indexers.

    Each chunk is extended by (block_size - stride) so that strided iteration inside
    gather_samples() always has enough data to fill complete blocks near chunk boundaries.
    """
    chunk_size = get_chunk_sizes(ds)

    dim_ranges = []
    for dim_name, chunk_dim_size in chunk_size:
        dim_ranges.append(range(0, ds.sizes[dim_name], chunk_dim_size))

    for offsets in itertools.product(*dim_ranges):
        dim_slices = {cs[0]: slice(offset, offset + cs[1])
                      for cs, offset in zip(chunk_size, offsets)}

        # Extend chunk boundaries so strided iteration never hits a partial block
        # at the seam between chunks. dim_ranges[i].stop equals the total dimension
        # size, so the check ensures we don't read beyond the array end.
        padding_time = block_size[0] - strides[0]
        padding_spatial = block_size[1] - strides[1]

        if dim_ranges[0].stop >= dim_slices['time'].stop + padding_time:
            dim_slices['time'] = np.s_[dim_slices['time'].start:dim_slices['time'].stop + padding_time]
        if dim_ranges[1].stop >= dim_slices['lat'].stop + padding_spatial:
            dim_slices['lat'] = np.s_[dim_slices['lat'].start:dim_slices['lat'].stop + padding_spatial]
        if dim_ranges[2].stop >= dim_slices['lon'].stop + padding_spatial:
            dim_slices['lon'] = np.s_[dim_slices['lon'].start:dim_slices['lon'].stop + padding_spatial]

        var_blocks = {}
        for var_name, var in ds.data_vars.items():
            indexers = {dim_name: dim_slice
                        for dim_name, dim_slice in dim_slices.items()
                        if dim_name in var.sizes}
            var_blocks[var_name] = var.isel(indexers).values

            time_coords = ds['time'][indexers['time'].start:indexers['time'].stop].values
            lat_coords = ds['lat'][indexers['lat'].start:indexers['lat'].stop].values
            lon_coords = ds['lon'][indexers['lon'].start:indexers['lon'].stop].values

            if 'coordinates' not in var_blocks:
                var_blocks['coordinates'] = {}
            var_blocks['coordinates'][var_name] = {'time': time_coords, 'lat': lat_coords, 'lon': lon_coords}

            if 'indexers' not in var_blocks:
                var_blocks['indexers'] = {}
            var_blocks['indexers'][var_name] = indexers

        yield var_blocks


def gather_samples(ds, var, path, block_size, strides, patch_overlap, aux_vars, max_gap_portion, syn_gaps=False, no_split=False, gap_donor_min_ratio=0.05, gap_donor_max_ratio=0.85):
    """Extract spatiotemporal patches from ds and write them to an HDF5 dataset file.

    Args:
        syn_gaps: If True, inject synthetic cloud gaps into the middle input time step
            (for supervised gap-filling training). If False, the input is used as-is
            (for reconstruction pre-training).
    """
    amount_channels = len(aux_vars) + 1

    shape_X = (block_size[0], block_size[1] + patch_overlap * 2, block_size[2] + patch_overlap * 2, amount_channels)
    shape_y = (1, block_size[1], block_size[2], 1)
    block_size_with_overlap = (block_size[0], block_size[1] + patch_overlap * 2, block_size[2] + patch_overlap * 2)
    middle_idx = int((block_size[0] - 1) * 0.5)

    with h5py.File(path + 'dataset.h5', 'w') as f:
        X_train = f.create_dataset('X_train', shape=(0,) + shape_X, maxshape=(None,) + shape_X, dtype='float32', chunks=True)
        y_train = f.create_dataset('y_train', shape=(0,) + shape_y, maxshape=(None,) + shape_y, dtype='float32', chunks=True)
        X_test  = f.create_dataset('X_test',  shape=(0,) + shape_X, maxshape=(None,) + shape_X, dtype='float32', chunks=True)
        y_test  = f.create_dataset('y_test',  shape=(0,) + shape_y, maxshape=(None,) + shape_y, dtype='float32', chunks=True)
        time_train = f.create_dataset('time_train', shape=(0, 2), maxshape=(None, 2), dtype=h5py.string_dtype(), chunks=True)
        time_test  = f.create_dataset('time_test',  shape=(0, 2), maxshape=(None, 2), dtype=h5py.string_dtype(), chunks=True)
        lat_train  = f.create_dataset('lat_train',  shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
        lat_test   = f.create_dataset('lat_test',   shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
        lon_train  = f.create_dataset('lon_train',  shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
        lon_test   = f.create_dataset('lon_test',   shape=(0,), maxshape=(None,), dtype='float32', chunks=True)

        num_samples_train = 0
        num_samples_test = 0
        batch_size = 100
        train_batch = BatchData()
        test_batch = BatchData()
        discarded_no_donor = 0

        with tqdm.tqdm(total=calculate_total_chunks(ds)) as pbar:
            for chunk in iter_data_var_blocks_with_indexers(ds, block_size_with_overlap, strides):
                pbar.update(1)

                data = chunk[var]
                if np.isnan(data).all():
                    continue

                if len(aux_vars) > 0:
                    aux_data = np.array([chunk[av['var']] for av in aux_vars if av['var'] in chunk])
                    aux_data = np.transpose(aux_data, (1, 2, 3, 0))

                coordinates = chunk['coordinates'][var]
                split_mask = chunk['split'] if not no_split else None

                # Preload the full time series for this chunk's spatial extent so donor
                # mask selection can be done in-memory without per-block zarr reads.
                if syn_gaps:
                    chunk_indexers = chunk['indexers'][var]
                    chunk_t_start = chunk_indexers['time'].start
                    chunk_all_times = ds[var].isel(
                        lat=chunk_indexers['lat'],
                        lon=chunk_indexers['lon']
                    ).values  # (n_all_times, H_chunk, W_chunk)

                stride_t, stride_lat, stride_lon = strides
                for t in range(0, data.shape[0], stride_t):
                    for i_lat in range(0, data.shape[1], stride_lat):
                        for i_lon in range(0, data.shape[2], stride_lon):

                            # Discard remainders that don't fill a complete block.
                            if not (t + block_size_with_overlap[0] <= data.shape[0]
                                    and i_lat + block_size_with_overlap[1] <= data.shape[1]
                                    and i_lon + block_size_with_overlap[2] <= data.shape[2]):
                                continue

                            block_X = np.copy(data[
                                t:t + block_size_with_overlap[0],
                                i_lat:i_lat + block_size_with_overlap[1],
                                i_lon:i_lon + block_size_with_overlap[2]
                            ])

                            # Target is the middle time step, cropped to remove overlap.
                            block_y = data[
                                t + middle_idx,
                                i_lat + patch_overlap:i_lat + patch_overlap + block_size[1],
                                i_lon + patch_overlap:i_lon + patch_overlap + block_size[2]
                            ]

                            if np.isnan(block_y).sum() / block_y.size > max_gap_portion:
                                continue

                            block_y = np.asarray(block_y)
                            if hasattr(block_y, "values"):
                                block_y = block_y.values
                            if isinstance(block_y, np.ma.MaskedArray):
                                block_y = block_y.filled(np.nan)
                            block_y = block_y.astype("float32")[np.newaxis, ..., np.newaxis]

                            block_X = np.expand_dims(block_X, axis=-1)
                            if len(aux_vars) > 0:
                                aux_block = np.copy(aux_data[
                                    t:t + block_size_with_overlap[0],
                                    i_lat:i_lat + block_size_with_overlap[1],
                                    i_lon:i_lon + block_size_with_overlap[2], :
                                ])
                                block_X = np.concatenate((block_X, aux_block), axis=-1)

                            if syn_gaps:
                                donor_mask = _fetch_donor_mask_series(
                                    chunk_all_times,
                                    i_lat, i_lat + block_size_with_overlap[1],
                                    i_lon, i_lon + block_size_with_overlap[2],
                                    chunk_t_start + t,
                                    block_size[0],
                                    min_gap_frac=gap_donor_min_ratio,
                                    max_gap_frac=gap_donor_max_ratio,
                                )
                                if donor_mask is None:
                                    discarded_no_donor += 1
                                    continue
                            else:
                                donor_mask = None

                            block_time = np.datetime_as_string(np.array([
                                coordinates['time'][t],
                                coordinates['time'][t + block_size[0] - 1]
                            ]))
                            block_lat = float(coordinates['lat'][i_lat + patch_overlap])
                            block_lon = float(coordinates['lon'][i_lon + patch_overlap])

                            if not no_split and split_mask[t, i_lat, i_lon]:
                                test_batch.append(block_X, block_y, block_time, block_lat, block_lon, donor_mask)
                                if len(test_batch) == batch_size:
                                    num_samples_test = _write_batch(
                                        X_test, y_test, time_test, lat_test, lon_test,
                                        test_batch, num_samples_test, shape_X, shape_y, patch_overlap, syn_gaps
                                    )
                            else:
                                train_batch.append(block_X, block_y, block_time, block_lat, block_lon, donor_mask)
                                if len(train_batch) == batch_size:
                                    num_samples_train = _write_batch(
                                        X_train, y_train, time_train, lat_train, lon_train,
                                        train_batch, num_samples_train, shape_X, shape_y, patch_overlap, syn_gaps
                                    )

            # Flush any remaining samples that didn't fill a full batch.
            if len(train_batch) > 0:
                num_samples_train = _write_batch(
                    X_train, y_train, time_train, lat_train, lon_train,
                    train_batch, num_samples_train, shape_X, shape_y, patch_overlap, syn_gaps
                )
            if len(test_batch) > 0:
                num_samples_test = _write_batch(
                    X_test, y_test, time_test, lat_test, lon_test,
                    test_batch, num_samples_test, shape_X, shape_y, patch_overlap, syn_gaps
                )

        if syn_gaps and discarded_no_donor > 0:
            print(f'  Discarded {discarded_no_donor} patches: no valid donor gap mask found.')


def create_dataset(var, path, block_size, strides, patch_overlap, aux_vars, max_gap_portion=0.2, split_ranges=None, syn_gaps=False, no_split=False, preprocessed_name='preprocessed.zarr', gap_donor_min_ratio=0.05, gap_donor_max_ratio=0.85, seed=None):
    """Build a train/test HDF5 dataset from a preprocessed zarr store.

    Args:
        syn_gaps: If True, inject synthetic cloud gaps into the middle input time step
            (supervised gap-filling). If False, use the data as-is (pre-training).
        no_split: If True, skip train/test splitting and write all samples to the train
            split. Intended for evaluation-only datasets (e.g. gap-filling eval) where
            a held-out test set is meaningless.
        seed: Optional seed mixed into the block-split hash (see assign_block_split).
            Pass the same seed to two datasets built over the same grid/block_size to
            guarantee an identical split.
    """
    ds = xr.open_zarr(path + preprocessed_name)

    chunk_size = tuple(chunk[0] for chunk in ds[var].chunks)
    print('Chunk Size: ' + str(chunk_size))

    if not no_split:
        if split_ranges:
            ds = assign_sloo_split(
                ds,
                block_size=[("time", block_size[0]), ("lat", block_size[1]), ("lon", block_size[2])],
                split_ranges=split_ranges
            )
        else:
            ds = assign_block_split(
                ds,
                block_size=[("time", block_size[0]), ("lat", block_size[1]), ("lon", block_size[2])],
                split=0.8,
                seed=seed
            )

    gather_samples(ds, var, path, block_size, strides, patch_overlap, aux_vars, max_gap_portion, syn_gaps=syn_gaps, no_split=no_split, gap_donor_min_ratio=gap_donor_min_ratio, gap_donor_max_ratio=gap_donor_max_ratio)
