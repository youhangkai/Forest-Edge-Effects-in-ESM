#!/usr/bin/env python3
"""
step2_surfdata_canopy.py

Aggregates 500m WGS84 canopy height (CONUS) to the 0.5° ELM/CLM NC grid
per forest PFT, then replaces MONTHLY_HEIGHT_TOP in the surface data file
for 3 scenarios.  Only PFTs 1–8 (trees) at CONUS cells are replaced;
all other variables, PFTs, and regions are kept identical to the template.

Inputs
------
  Canopy_Height/processed/canopy_height_500m_{suffix}.tif  (3 scenarios)
  PFT/ELM_PFT_2020-WGS84-merged.tif
  surfdata_0.5x0.5_simyr2010_c251122.nc  (template)

Outputs (3 new surfdata files)
-------
  surfdata_revised/surfdata_0.5x0.5_CH_v1_all_forests.nc
  surfdata_revised/surfdata_0.5x0.5_CH_v2_exterior_forests.nc
  surfdata_revised/surfdata_0.5x0.5_CH_v3_interior_forests.nc

9-scenario run matrix (surfdata × LAI stream)
----------------------------------------------
              | CH only            | LAI only            | Both
  area-wtd    | CH_v1 + LAI_orig   | CH_orig + LAI_v1    | CH_v1 + LAI_v1
  interior    | CH_v3 + LAI_orig   | CH_orig + LAI_v3    | CH_v3 + LAI_v3
  exterior    | CH_v2 + LAI_orig   | CH_orig + LAI_v2    | CH_v2 + LAI_v2

LAI stream files:
  original : MODISPFTLAI_0.5x0.5_c140711.nc
  v1       : LAI_revised/MODISPFTLAI_0.5x0.5_v1_all_forests.nc
  v2       : LAI_revised/MODISPFTLAI_0.5x0.5_v2_exterior_forests.nc
  v3       : LAI_revised/MODISPFTLAI_0.5x0.5_v3_interior_forests.nc
"""

import numpy as np
import rasterio
from scipy.io import netcdf_file
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
CH_DIR    = "/mnt/cephfs-mount/hangkai/Canopy_Height/processed"
PFT_FILE  = "/mnt/cephfs-mount/hangkai/PFT/ELM_PFT_2020-WGS84-merged.tif"
SURF_TMPL = "/mnt/cephfs-mount/hangkai/surfdata_0.5x0.5_simyr2010_c251122.nc"
OUT_DIR   = "/mnt/cephfs-mount/hangkai/surfdata_revised"

NC_NLAT = 360
NC_NLON = 720
NC_DLAT = 0.5
NC_DLON = 0.5

N_FOREST_PFTS = 8    # PFTs 1–8 in surfdata (indices 1–8 in 17-element lsmpft dim)
FILL_VALUE    = -9999.0

SCENARIOS = [
    ("v1_all_forests",      "canopy_height_500m_v1_all_forests.tif"),
    ("v2_exterior_forests", "canopy_height_500m_v2_exterior_forests.tif"),
    ("v3_interior_forests", "canopy_height_500m_v3_interior_forests.tif"),
]


# ── 500m PFT-grid CH → 0.5° NC grid, per forest PFT ──────────────────────────
def aggregate_to_nc(ch_conus, pft_conus, pft_x0, pft_y0, pft_cw, pft_ch,
                    pft_bounds, pft_res):
    """
    Count-weighted aggregation from 500m CONUS pixels to the 0.5° NC grid,
    stratified by forest PFT.

    Returns
    -------
    ch_nc : float64 (N_FOREST_PFTS, NC_NLAT, NC_NLON)
            Mean canopy height per PFT per 0.5° cell; FILL_VALUE where empty.
    """
    # Pixel-centre lon/lat for the CONUS PFT sub-grid (N→S for lat)
    lon = pft_bounds.left + (pft_x0 + np.arange(pft_cw) + 0.5) * pft_res
    lat = pft_bounds.top  - (pft_y0 + np.arange(pft_ch) + 0.5) * pft_res

    # NC grid column / row (ascending S→N in NC lat convention)
    nc_col = np.floor((lon + 180.0) / NC_DLON).astype(np.int32)
    nc_row = np.floor((lat +  90.0) / NC_DLAT).astype(np.int32)
    np.clip(nc_col, 0, NC_NLON - 1, out=nc_col)
    np.clip(nc_row, 0, NC_NLAT - 1, out=nc_row)

    total_bins = N_FOREST_PFTS * NC_NLAT * NC_NLON
    wsum   = np.zeros(total_bins, np.float64)
    wtotal = np.zeros(total_bins, np.int64)

    # Valid: CH > fill AND PFT is a forest type (1–8)
    valid = (
        (ch_conus  > FILL_VALUE + 1) &
        (pft_conus >= 1) & (pft_conus <= N_FOREST_PFTS)
    )
    cy_arr, cx_arr = np.where(valid)

    if cy_arr.size > 0:
        pft_idx  = pft_conus[cy_arr, cx_arr].astype(np.int32) - 1   # 0-based
        col_nc   = nc_col[cx_arr]
        row_nc   = nc_row[cy_arr]
        lin      = pft_idx * (NC_NLAT * NC_NLON) + row_nc * NC_NLON + col_nc
        vals     = ch_conus[cy_arr, cx_arr].astype(np.float64)

        wsum   += np.bincount(lin, weights=vals, minlength=total_bins)
        wtotal += np.bincount(lin,               minlength=total_bins).astype(np.int64)

    ch_nc = np.full((N_FOREST_PFTS, NC_NLAT, NC_NLON), FILL_VALUE, np.float64)
    wt = wtotal.reshape(N_FOREST_PFTS, NC_NLAT, NC_NLON)
    ws = wsum  .reshape(N_FOREST_PFTS, NC_NLAT, NC_NLON)
    ok = wt > 0
    ch_nc[ok] = ws[ok] / wt[ok]

    return ch_nc


# ── Write modified surfdata NC file ───────────────────────────────────────────
def write_surfdata(out_path, tmpl_vars, tmpl_attrs, htop_new):
    """
    Copy template surfdata and replace MONTHLY_HEIGHT_TOP for PFTs 1–8
    at CONUS cells where htop_new has valid data.

    htop_new : float64 (N_FOREST_PFTS, NC_NLAT, NC_NLON)
               Aggregated canopy height; FILL_VALUE where no replacement.
    """
    print(f"  Writing {Path(out_path).name} …", flush=True)

    with netcdf_file(str(out_path), 'w', version=2) as f:
        # Global attributes
        for attr, val in tmpl_attrs.items():
            setattr(f, attr, val)

        # Dimensions — unlimited (None) must be created first (NetCDF-3 rule)
        dims = tmpl_vars['_dimensions']
        for dim_name, dim_size in dims.items():
            if dim_size is None:
                f.createDimension(dim_name, dim_size)
        for dim_name, dim_size in dims.items():
            if dim_size is not None:
                f.createDimension(dim_name, dim_size)

        # Variables
        for vname, vinfo in tmpl_vars.items():
            if vname == '_dimensions':
                continue
            typecode, dims, vattrs, vdata = vinfo
            var = f.createVariable(vname, typecode, dims)
            for attr, val in vattrs.items():
                setattr(var, attr, val)

            if vname == 'MONTHLY_HEIGHT_TOP':
                # shape: (12, 17, 360, 720)
                # PFT indices 1–8 are forest; replace where htop_new is valid
                out = vdata.copy()
                for pft_i in range(N_FOREST_PFTS):
                    surf_pft_idx = pft_i + 1    # 1-based in the 17-element dim
                    new_htop = htop_new[pft_i]  # (NC_NLAT, NC_NLON)
                    valid    = new_htop > FILL_VALUE + 1
                    # Apply same replacement for all 12 months (height is static)
                    out[:, surf_pft_idx, :, :] = np.where(
                        valid[np.newaxis, :, :],
                        new_htop[np.newaxis, :, :].astype(out.dtype),
                        out[:, surf_pft_idx, :, :]
                    )
                var[:] = out

            else:
                if vdata is not None:
                    if vdata.ndim == 0:
                        var.data[()] = vdata
                    else:
                        var[:] = vdata

    size_mb = Path(out_path).stat().st_size / 1024**2
    print(f"  Done: {Path(out_path).name}  ({size_mb:.0f} MB)", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    print("=" * 64)
    print("STEP 2 — Surface Data Canopy Height Replacement")
    print("=" * 64)
    print(f"Template : {SURF_TMPL}")
    print(f"CH dir   : {CH_DIR}")
    print(f"Output   : {OUT_DIR}")

    # 1. Load PFT (windowed CONUS clip aligned with CH files)
    print("\nLoading PFT (CONUS clip) …")
    with rasterio.open(f"{CH_DIR}/canopy_height_500m_v1_all_forests.tif") as ch_src:
        ch_bounds = ch_src.bounds

    with rasterio.open(PFT_FILE) as src:
        pft_bounds = src.bounds
        pft_res    = src.res[0]
        window     = src.window(*ch_bounds).round_offsets().round_lengths()
        pft_x0     = int(window.col_off)
        pft_y0     = int(window.row_off)
        pft_cw     = int(window.width)
        pft_ch     = int(window.height)
        pft_conus  = src.read(1, window=window).astype(np.uint8)

    print(f"  CONUS: {pft_cw}×{pft_ch} cells  (offset col={pft_x0}, row={pft_y0})")
    forest_cells = ((pft_conus >= 1) & (pft_conus <= N_FOREST_PFTS)).sum()
    print(f"  Forest cells (PFT 1–8): {forest_cells:,}")

    # 2. Load surfdata template into memory (all variables)
    print("\nLoading surfdata template …")
    t_load = datetime.now()
    tmpl_vars  = {}
    tmpl_attrs = {}
    with netcdf_file(SURF_TMPL, 'r', mmap=False) as ds:
        for attr in ds._attributes:
            tmpl_attrs[attr] = getattr(ds, attr)
        tmpl_vars['_dimensions'] = dict(ds.dimensions)
        for vname, var in ds.variables.items():
            typecode = var.typecode()
            vdims    = var.dimensions
            vattrs   = {a: getattr(var, a) for a in var._attributes}
            vdata    = var.data.copy()
            tmpl_vars[vname] = (typecode, vdims, vattrs, vdata)
    print(f"  Loaded {len(tmpl_vars)-1} variables in {datetime.now()-t_load}")

    # Print original HTOP stats for reference
    orig_htop = tmpl_vars['MONTHLY_HEIGHT_TOP'][3]   # vdata
    print("  Original MONTHLY_HEIGHT_TOP (month 1, PFTs 1–8):")
    for p in range(N_FOREST_PFTS):
        vals = orig_htop[0, p + 1, :, :]
        nz = vals[vals > 0]
        print(f"    PFT {p+1}: mean(>0)={nz.mean():.2f}m  max={vals.max():.2f}m")

    # 3. Process each scenario
    for suffix, ch_fname in SCENARIOS:
        print(f"\n{'─'*60}")
        print(f"Scenario: {suffix}")

        # Load 500m CH (CONUS extent)
        with rasterio.open(Path(CH_DIR) / ch_fname) as src:
            ch_conus = src.read(1).astype(np.float32)
        valid_n = (ch_conus > FILL_VALUE + 1).sum()
        print(f"  CH valid 500m cells: {valid_n:,}  "
              f"mean={ch_conus[ch_conus > FILL_VALUE+1].mean():.2f}m")

        # Aggregate to 0.5° per forest PFT
        print("  Aggregating to 0.5° NC grid …")
        ch_nc = aggregate_to_nc(ch_conus, pft_conus,
                                pft_x0, pft_y0, pft_cw, pft_ch,
                                pft_bounds, pft_res)

        print("  Aggregated heights per PFT:")
        for p in range(N_FOREST_PFTS):
            m = ch_nc[p] > FILL_VALUE + 1
            if m.any():
                print(f"    PFT {p+1}: {m.sum():>6,} NC cells, "
                      f"mean={ch_nc[p][m].mean():.2f}m  "
                      f"max={ch_nc[p][m].max():.2f}m")

        # Write modified surfdata
        out_path = Path(OUT_DIR) / f"surfdata_0.5x0.5_CH_{suffix}.nc"
        write_surfdata(out_path, tmpl_vars, tmpl_attrs, ch_nc)

    print(f"\n{'='*64}")
    print(f"Done!  Total time: {datetime.now()-t0}")
    print(f"Output: {OUT_DIR}")

    print("""
9-Scenario Run Matrix
=====================
              | CH only             | LAI only            | Both
  area-wtd   | CH_v1 + LAI_orig    | CH_orig + LAI_v1    | CH_v1 + LAI_v1
  interior   | CH_v3 + LAI_orig    | CH_orig + LAI_v3    | CH_v3 + LAI_v3
  exterior   | CH_v2 + LAI_orig    | CH_orig + LAI_v2    | CH_v2 + LAI_v2

surfdata files:
  original : surfdata_0.5x0.5_simyr2010_c251122.nc
  CH_v1    : surfdata_revised/surfdata_0.5x0.5_CH_v1_all_forests.nc
  CH_v2    : surfdata_revised/surfdata_0.5x0.5_CH_v2_exterior_forests.nc
  CH_v3    : surfdata_revised/surfdata_0.5x0.5_CH_v3_interior_forests.nc

LAI stream files:
  original : MODISPFTLAI_0.5x0.5_c140711.nc
  LAI_v1   : LAI_revised/MODISPFTLAI_0.5x0.5_v1_all_forests.nc
  LAI_v2   : LAI_revised/MODISPFTLAI_0.5x0.5_v2_exterior_forests.nc
  LAI_v3   : LAI_revised/MODISPFTLAI_0.5x0.5_v3_interior_forests.nc
""")


if __name__ == "__main__":
    main()
