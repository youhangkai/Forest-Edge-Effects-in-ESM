#!/usr/bin/env python3
"""
Step 3 — Revise MODIS PFT LAI NetCDF (0.5°) with CONUS forest LAI.

Reads MODISPFTLAI_0.5x0.5_c140711.nc and replaces forest LAI (PFT 1–8) in
the CONUS region with values derived from Step 1 (500m WGS84) outputs,
aggregated to the NC file's 0.5° × 0.5° grid using count-weighted averaging.

Three output files (one per scenario):
    LAI_revised_v3_interior_forests.nc  — interior forest only  (depth == 5)
    LAI_revised_v2_exterior_forests.nc  — exterior forest only  (depth 1–4)
    LAI_revised_v1_all_forests.nc       — area-weighted all forest (depth 1–5)

Replacement rules:
    • CONUS cell with valid step1 data  → replaced with aggregated 0.5° LAI
    • CONUS cell with no valid step1 data → original NC value kept unchanged
    • Outside CONUS / fill values        → original NC value kept unchanged

Note: The NC time axis covers Jan 2001 – Dec 2013 (156 months).
      Step1 data for 2013-Nov and 2013-Dec is absent from the source, so
      those two time steps retain their original NC values.
"""

import os
import numpy as np
import rasterio
from scipy.io import netcdf_file
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
STEP1_DIR   = "/mnt/cephfs-mount/hangkai/LAI_500m"
PFT_DIR     = "/mnt/cephfs-mount/hangkai/PFT"
NC_TEMPLATE = "/mnt/cephfs-mount/hangkai/MODISPFTLAI_0.5x0.5_c140711.nc"
OUTPUT_DIR  = "/mnt/cephfs-mount/hangkai/LAI_revised"

# NC file temporal coverage: 156 months = Jan 2001 – Dec 2013
N_TIMES = 156
MONTHS  = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# PFTs 1–8 are tree/forest types in the ELM/CLM classification
N_FOREST_PFTS = 8
FOREST_PFT_VARS = [f"LAI_{p}" for p in range(1, N_FOREST_PFTS + 1)]

# NC grid: 0.5° global, lat S→N, lon W→E
NC_NLAT = 360   # lat: -89.75 … 89.75  (row 0 = south)
NC_NLON = 720   # lon: -179.75 … 179.75 (col 0 = west)
NC_DLAT = 0.5
NC_DLON = 0.5

FILL_VALUE = np.float32(-9999.0)

# Step-1 band indices (0-indexed after src.read())
BAND_INTERIOR_MEAN  = 0
BAND_EXTERIOR_MEAN  = 1
BAND_WEIGHTED_MEAN  = 2
BAND_INTERIOR_COUNT = 3
BAND_EXTERIOR_COUNT = 4
BAND_TOTAL_COUNT    = 5

# Scenarios: (output_suffix, mean_band, count_band)
SCENARIOS = [
    ("v1_all_forests",      BAND_WEIGHTED_MEAN,  BAND_TOTAL_COUNT),
    ("v2_exterior_forests", BAND_EXTERIOR_MEAN,  BAND_EXTERIOR_COUNT),
    ("v3_interior_forests", BAND_INTERIOR_MEAN,  BAND_INTERIOR_COUNT),
]

# ── Module-level caches ────────────────────────────────────────────────────────
_pft_cache  = {}   # (year, s1_w, s1_h, bounds_str) → uint8 CONUS-clipped PFT
_grid_cache = {}   # (s1_w, s1_h, bounds_str) → (cx_col, cy_row, conus_box)


# ── 500m → 0.5° NC-grid index mapping ────────────────────────────────────────
def compute_nc_indices(bounds, width, height):
    """
    Map each 500m CONUS pixel to its 0.5° NC grid cell.

    NC lat convention: ascending (S → N), row 0 = lat –89.75
    NC lon convention: ascending (W → E), col 0 = lon –179.75

    Returns
    -------
    cx_col    : int32 (width,)   — NC lon column for each CONUS column
    cy_row    : int32 (height,)  — NC lat row    for each CONUS row
    conus_box : (x0, y0, cw, ch) — CONUS sub-rectangle in the NC grid
                (x0=leftmost NC col, y0=southernmost NC row)
    """
    lon = np.linspace(bounds.left,  bounds.right,  width)
    lat = np.linspace(bounds.top,   bounds.bottom, height)   # N → S (GeoTIFF)

    # NC col: floor((lon + 180) / 0.5)
    cx = np.floor((lon + 180.0) / NC_DLON).astype(np.int32)
    # NC row: floor((lat + 90)  / 0.5)  — ascending from south
    cy = np.floor((lat +  90.0) / NC_DLAT).astype(np.int32)

    np.clip(cx, 0, NC_NLON - 1, out=cx)
    np.clip(cy, 0, NC_NLAT - 1, out=cy)

    x0, x1 = int(cx.min()), int(cx.max())
    y0, y1 = int(cy.min()), int(cy.max())
    conus_box = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    return cx, cy, conus_box


# ── Count-weighted aggregation to 0.5° grid ──────────────────────────────────
def aggregate_to_nc(lai_band, count_band, pft_data, cx_col, cy_row, conus_box):
    """
    Aggregate 500m LAI + pixel counts to the 0.5° NC CONUS sub-grid for all
    8 forest PFTs simultaneously, using count-weighted averaging:

        nc_LAI = Σ(lai_mean_i × pixel_count_i) / Σ(pixel_count_i)

    Parameters
    ----------
    lai_band   : float32 (h, w) — Step1 LAI mean per 500m cell
    count_band : float32 (h, w) — Step1 valid 30m pixel count per 500m cell
    pft_data   : uint8   (h, w) — PFT values 1–8 for same CONUS grid
    cx_col     : int32   (w,)   — NC lon column per CONUS column
    cy_row     : int32   (h,)   — NC lat row   per CONUS row (ascending)
    conus_box  : (x0, y0, cw, ch)

    Returns
    -------
    result : float32 (N_FOREST_PFTS, ch, cw) — FILL_VALUE where no valid data
    """
    x0, y0, cw, ch = conus_box
    n_conus    = cw * ch
    total_bins = N_FOREST_PFTS * n_conus

    valid = (
        (lai_band   > 0) & (lai_band   < 100) &
        (count_band > 0) &
        (pft_data   >= 1) & (pft_data  <= N_FOREST_PFTS)
    )

    rr, cc = np.where(valid)
    wsum   = np.zeros(total_bins, np.float64)
    wtotal = np.zeros(total_bins, np.float64)

    if rr.size > 0:
        pft_idx  = pft_data[rr, cc].astype(np.int32) - 1   # 0 … 7
        cx_local = cx_col[cc].astype(np.int32) - x0
        cy_local = cy_row[rr].astype(np.int32) - y0         # 0 … ch-1 (S→N)
        lin_loc  = cy_local * cw + cx_local
        combined = pft_idx * n_conus + lin_loc

        w   = count_band[rr, cc].astype(np.float64)
        lai = lai_band  [rr, cc].astype(np.float64)

        wsum   += np.bincount(combined, weights=lai * w, minlength=total_bins)
        wtotal += np.bincount(combined, weights=w,       minlength=total_bins)

    result = np.full((N_FOREST_PFTS, ch, cw), FILL_VALUE, np.float32)
    ws = wsum .reshape(N_FOREST_PFTS, ch, cw)
    wt = wtotal.reshape(N_FOREST_PFTS, ch, cw)
    ok = wt > 0
    result[ok] = (ws[ok] / wt[ok]).astype(np.float32)

    return result   # (N_FOREST_PFTS, ch, cw)


# ── Write output NetCDF (copy template, replace forest LAI) ───────────────────
def write_nc(out_path, template_vars, template_attrs, modified_lai, conus_box,
             orig_lai_arrays):
    """
    Write a new NetCDF-3 (64-bit offset) file, copying all template variables
    and replacing the CONUS forest LAI with modified_lai where valid.

    Parameters
    ----------
    out_path         : str/Path
    template_vars    : dict  varname → (typecode, dims, attrs, data)
    template_attrs   : dict  global attributes
    modified_lai     : float32 (N_FOREST_PFTS, N_TIMES, conus_ch, conus_cw)
    conus_box        : (x0, y0, cw, ch) in NC lon/lat space
    orig_lai_arrays  : list of N_FOREST_PFTS float32 arrays (156, 360, 720)
    """
    x0, y0, cw, ch = conus_box
    print(f"  Writing {out_path} …", flush=True)

    with netcdf_file(str(out_path), 'w', version=2) as f:
        # ── Global attributes ──────────────────────────────────────────────
        for attr, val in template_attrs.items():
            setattr(f, attr, val)

        # ── Dimensions ────────────────────────────────────────────────────
        for dim_name, dim_size in template_vars['_dimensions'].items():
            f.createDimension(dim_name, dim_size)

        # ── Variables ─────────────────────────────────────────────────────
        for vname, vinfo in template_vars.items():
            if vname == '_dimensions':
                continue
            typecode, dims, vattrs, vdata = vinfo

            var = f.createVariable(vname, typecode, dims)
            for attr, val in vattrs.items():
                setattr(var, attr, val)

            if vname in FOREST_PFT_VARS:
                pft_i = int(vname.split('_')[1]) - 1   # 0-based PFT index
                # Start with original data
                out_data = orig_lai_arrays[pft_i].copy()
                # Replace CONUS cells where modified value is valid
                new_conus = modified_lai[pft_i]          # (156, ch, cw)
                valid     = new_conus > (FILL_VALUE + 1)  # (156, ch, cw)
                out_data[:, y0:y0 + ch, x0:x0 + cw][valid] = new_conus[valid]
                var.data[:] = out_data.astype(np.float32)
            else:
                if vdata is not None:
                    var.data[:] = vdata

    print(f"  Done: {out_path}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("STEP 3: Revise MODIS PFT LAI NC with CONUS forest LAI (0.5°)")
    print("=" * 64)
    print(f"Template: {NC_TEMPLATE}")
    print(f"Step1 dir: {STEP1_DIR}")
    print(f"Output:   {OUTPUT_DIR}")
    print()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    # ── 1. Read template NC file ──────────────────────────────────────────
    print("Loading template NC file …")
    template_vars  = {}
    template_attrs = {}
    orig_lai_arrays = []

    with netcdf_file(NC_TEMPLATE, 'r', mmap=False) as ds:
        # Global attributes
        for attr in ds._attributes:
            template_attrs[attr] = getattr(ds, attr)

        # Dimensions (scipy stores sizes directly as ints, not dimension objects)
        dims_sizes = dict(ds.dimensions)
        template_vars['_dimensions'] = dims_sizes

        # Variables
        for vname, var in ds.variables.items():
            typecode = var.typecode()
            vdims    = var.dimensions
            vattrs   = {}
            for attr in var._attributes:
                vattrs[attr] = getattr(var, attr)
            # Forest LAI variables are stored separately in orig_lai_arrays
            # to avoid double-loading; set vdata=None for them here
            if vname in FOREST_PFT_VARS:
                vdata = None
            else:
                vdata = var.data.copy()
            template_vars[vname] = (typecode, vdims, vattrs, vdata)

        # Keep original forest LAI arrays for replacement later
        for p in range(1, N_FOREST_PFTS + 1):
            orig_lai_arrays.append(
                ds.variables[f'LAI_{p}'].data.copy().astype(np.float32)
            )

    print(f"  Template loaded: {len(template_vars)-1} variables, "
          f"{N_TIMES} time steps, grid {NC_NLON}×{NC_NLAT}")

    # ── 2. Find grid mapping from first available Step1 file ─────────────
    print("Computing 500m → 0.5° grid mapping …")
    s1_sample = None
    for year in range(2001, 2014):
        f = f"{STEP1_DIR}/{year}/{year}_Jan_forest_lai_500m.tif"
        if os.path.exists(f):
            s1_sample = f
            break
    if s1_sample is None:
        raise RuntimeError("No Step1 files found — run step1 first.")

    with rasterio.open(s1_sample) as s1:
        s1_w, s1_h = s1.width, s1.height
        s1_bounds  = s1.bounds

    grid_key = (s1_w, s1_h, str(s1_bounds))
    cx_col, cy_row, conus_box = compute_nc_indices(s1_bounds, s1_w, s1_h)
    _grid_cache[grid_key] = (cx_col, cy_row, conus_box)
    x0, y0, cw, ch = conus_box
    print(f"  CONUS in NC grid: lon cols {x0}–{x0+cw-1}, lat rows {y0}–{y0+ch-1}"
          f"  ({cw}×{ch} = {cw*ch} cells  vs {NC_NLON*NC_NLAT} global)")

    # ── 3. Aggregate all 156 months for all 3 scenarios simultaneously ────
    # replacement[scenario, pft_idx, t, cy_local, cx_local]
    replacement = np.full((len(SCENARIOS), N_FOREST_PFTS, N_TIMES, ch, cw),
                          FILL_VALUE, np.float32)

    print(f"\nAggregating {N_TIMES} months to 0.5° grid …")
    pbar = tqdm(range(N_TIMES), unit="month", dynamic_ncols=True)
    for t in pbar:
        year       = 2001 + t // 12
        month      = MONTHS[t % 12]
        step1_file = f"{STEP1_DIR}/{year}/{year}_{month}_forest_lai_500m.tif"

        if not os.path.exists(step1_file):
            pbar.write(f"  WARNING: missing {step1_file} — keeping original NC values")
            continue

        # Load PFT clipped to CONUS (cached per year)
        pft_key = (year, s1_w, s1_h, str(s1_bounds))
        if pft_key not in _pft_cache:
            pft_file = f"{PFT_DIR}/ELM_PFT_{year}-WGS84-merged.tif"
            with rasterio.open(pft_file) as src:
                window = src.window(*s1_bounds)
                window = window.round_offsets().round_lengths()
                _pft_cache[pft_key] = src.read(1, window=window).astype(np.uint8)
        pft_data = _pft_cache[pft_key]

        # Load all 6 Step1 bands (CONUS, ~390 MB compressed → ~2.4 GB in RAM)
        with rasterio.open(step1_file) as src:
            bands = src.read().astype(np.float32)   # (6, s1_h, s1_w)

        # Aggregate for all 3 scenarios
        for s_i, (_, mean_band, count_band) in enumerate(SCENARIOS):
            agg = aggregate_to_nc(
                bands[mean_band], bands[count_band], pft_data,
                cx_col, cy_row, conus_box
            )
            replacement[s_i, :, t, :, :] = agg   # (N_FOREST_PFTS, ch, cw)

        pbar.set_postfix(year=year, month=month)

    pbar.close()
    print(f"\nAggregation done in {datetime.now() - t0}")

    # ── 4. Write 3 output NC files ─────────────────────────────────────────
    print("\nWriting output NC files …")
    for s_i, (suffix, _, _) in enumerate(SCENARIOS):
        out_path = Path(OUTPUT_DIR) / f"MODISPFTLAI_0.5x0.5_{suffix}.nc"
        write_nc(
            out_path,
            template_vars,
            template_attrs,
            replacement[s_i],    # (N_FOREST_PFTS, N_TIMES, ch, cw)
            conus_box,
            orig_lai_arrays,
        )

    print(f"\nDone!  Total time: {datetime.now() - t0}")
    print(f"Output: {OUTPUT_DIR}")
    print("Files:")
    for suffix, _, _ in SCENARIOS:
        print(f"  MODISPFTLAI_0.5x0.5_{suffix}.nc")


if __name__ == "__main__":
    main()
