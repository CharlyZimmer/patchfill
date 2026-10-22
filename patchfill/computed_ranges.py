"""
Lookup-backed computed_range cache for ESDC variables.

Provides ensure_computed_ranges(), which is the main entry point used by
create_esdc_dataset.py. Results are stored in a JSON file so they don't
need to be recomputed across dataset creation runs.

Lookup semantics (one entry per variable):
- If the stored entry covers a time span with >= as many days as the
  requested span, the stored values are reused.
- If the requested span is longer, new values are computed from the ESDC
  and the entry is overwritten.
"""
import itertools
import json
import os
import time
from datetime import datetime

import dask
import fsspec
from fsspec.config import conf as _fsspec_conf
import numpy as np
import xarray as xr
import zarr

# Limit concurrent HTTP chunk fetches to avoid 429 rate limits from the ESDC server.
# _fsspec_conf covers fsspec's own batching; zarr.config (v3 only) covers zarr's async layer.
_fsspec_conf["nofiles_gather_batch_size"] = 1
_fsspec_conf["gather_batch_size"] = 1
if hasattr(zarr, "config"):  # zarr v3+
    zarr.config.set({"async.concurrency": 1})

ESDC_URL = (
    "https://data.rsc4earth.de/download/EarthSystemDataCube/v2.1.1/"
    "esdc-8d-0.083deg-184x270x270-2.1.1.zarr"
)


def _span_days(time_start: str, time_end: str) -> int:
    fmt = "%Y-%m-%d"
    return (datetime.strptime(time_end, fmt) - datetime.strptime(time_start, fmt)).days


def load_computed_ranges(lookup_path: str) -> dict:
    if not os.path.exists(lookup_path):
        return {}
    with open(lookup_path) as f:
        return json.load(f)


def save_computed_ranges(lookup_path: str, ranges: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(lookup_path)), exist_ok=True)
    with open(lookup_path, "w") as f:
        json.dump(ranges, f, indent=2)


def check_time_span_validity(ds: xr.Dataset, var: str, time_start: str, time_end: str) -> None:
    """Raise ValueError if the first or last time step of var is entirely NaN."""
    da = ds[var]
    for label, idx in [("first", 0), ("last", -1)]:
        frame = da.isel(time=idx)
        with dask.config.set(scheduler="synchronous"):
            all_nan = bool(frame.isnull().all().compute())
        if all_nan:
            ts = str(da.time.values[idx])[:10]
            raise ValueError(
                f"{var}: {label} time step ({ts}) is entirely NaN "
                f"for the requested span {time_start} → {time_end}."
            )


def _chunk_slices(da: xr.DataArray) -> list:
    """Return one isel-kwargs dict per zarr chunk, derived from da.chunks.

    xr.open_zarr aligns dask chunks to zarr storage chunks by default, so each
    da.isel(one_slice).values call triggers exactly one HTTP GET.
    """
    dim_slices = []
    for dim, chunk_sizes in zip(da.dims, da.chunks):
        boundaries = [0]
        for sz in chunk_sizes:
            boundaries.append(boundaries[-1] + sz)
        dim_slices.append(
            [(dim, slice(boundaries[i], boundaries[i + 1])) for i in range(len(boundaries) - 1)]
        )
    return [
        {dim: sl for dim, sl in combo}
        for combo in itertools.product(*dim_slices)
    ]


def compute_var_range(ds: xr.Dataset, var: str, inter_chunk_sleep: float = 2.0) -> tuple:
    """Return (min, max) for var by scanning one zarr chunk at a time.

    Each chunk-aligned isel().values call triggers exactly one HTTP GET,
    preventing the concurrent fetches that cause 429 rate-limit errors.
    """
    da = ds[var]
    slices = _chunk_slices(da)
    total = len(slices)
    print(f"  Scanning {var} ({total} chunks) ...")
    global_min = np.inf
    global_max = -np.inf

    for idx, indexers in enumerate(slices):
        chunk = da.isel(indexers).values
        valid = chunk[np.isfinite(chunk)]
        if valid.size > 0:
            local_min = float(valid.min())
            local_max = float(valid.max())
            if local_min < global_min:
                global_min = local_min
            if local_max > global_max:
                global_max = local_max
        if (idx + 1) % 16 == 0 or idx == total - 1:
            print(f"    chunk {idx + 1}/{total}  running [{global_min:.4g}, {global_max:.4g}]", flush=True)
        if idx < total - 1:
            time.sleep(inter_chunk_sleep)

    if not np.isfinite(global_min):
        raise ValueError(f"{var}: no finite values found across all chunks.")
    print(f"  {var}: min={global_min:.4f}, max={global_max:.4f}")
    return global_min, global_max


def ensure_computed_ranges(
    vars_to_lookup: list,
    time_start: str,
    time_end: str,
    lookup_path: str,
    check_lst_validity: bool = True,
) -> dict:
    """Return {var: [min, max]} for every variable in vars_to_lookup.

    Reuses cached values when the stored time span is at least as long as
    the requested span. Computes and caches missing or shorter-span entries.

    Args:
        vars_to_lookup: Variable names that need computed_ranges.
        time_start: ISO date string for the start of the requested time span.
        time_end: ISO date string for the end of the requested time span.
        lookup_path: Path to the JSON lookup file.
        check_lst_validity: If True, verify that the first and last time steps
            of land_surface_temperature are non-empty before computing ranges.
    """
    ranges = load_computed_ranges(lookup_path)
    requested_days = _span_days(time_start, time_end)

    need_compute = [
        v for v in vars_to_lookup
        if v not in ranges
        or _span_days(ranges[v]["time_start"], ranges[v]["time_end"]) < requested_days
    ]

    result = {
        v: ranges[v]["computed_range"]
        for v in vars_to_lookup
        if v not in need_compute
    }

    if not need_compute:
        print(f"Computed ranges loaded from lookup for: {vars_to_lookup}")
        return result

    print(f"Computed ranges loaded from lookup for: {[v for v in vars_to_lookup if v not in need_compute]}")
    print(f"Computing ranges from ESDC for: {need_compute}")

    fs = fsspec.filesystem("https")
    ds = xr.open_zarr(fs.get_mapper(ESDC_URL), consolidated=True)
    ds = ds[need_compute].sel(time=slice(time_start, time_end))

    if check_lst_validity and "land_surface_temperature" in need_compute:
        check_time_span_validity(ds, "land_surface_temperature", time_start, time_end)

    for i, v in enumerate(need_compute):
        if i > 0:
            time.sleep(60.0)
        min_val, max_val = compute_var_range(ds, v)
        ranges[v] = {
            "time_start": time_start,
            "time_end": time_end,
            "computed_range": [min_val, max_val],
        }
        result[v] = [min_val, max_val]
        save_computed_ranges(lookup_path, ranges)
        print(f"Computed ranges saved to {lookup_path}")

    return result
