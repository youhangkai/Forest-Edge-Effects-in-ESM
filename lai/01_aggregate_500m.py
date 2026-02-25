#!/usr/bin/env python3
"""
Step 1 — 30m LAI × Forest Depth → 500m WGS84 forest LAI (CONUS-clipped).

Aggregates native 30m Albers LAI to the MODIS PFT grid (500m WGS84),
computing three LAI categories per grid cell:

    Band 1 — Interior LAI mean   : mean LAI where forest depth == 5
    Band 2 — Exterior LAI mean   : mean LAI where forest depth 1–4
    Band 3 — Weighted LAI mean   : mean LAI of all forest pixels (depth 1–5),
                                   naturally weighted by interior/exterior pixel count
    Band 4 — Interior pixel count: number of valid 30m pixels with depth == 5
    Band 5 — Exterior pixel count: number of valid 30m pixels with depth 1–4
    Band 6 — Total forest count  : number of valid 30m pixels with depth 1–5

IMPORTANT — CONUS clipping:
    The PFT file may cover the whole globe (hundreds of millions of cells).
    The LAI/Depth data covers only CONUS.  We therefore:
      1. Compute which rectangle of the global PFT grid the CONUS data touches.
      2. Do ALL computation in that CONUS-local PFT sub-grid.
      3. Write output files at the CONUS extent only (not the full global PFT grid).
    This keeps both memory and file sizes proportional to CONUS, not the globe.

Pixel counts (bands 4–6) enable proper count-weighted aggregation when
upscaling to any coarser resolution (0.5°, 1°, etc.) in the future:
    upscaled_LAI = Σ(mean_LAI_i × count_i) / Σ(count_i)

Output:  LAI_500m/{year}/{year}_{month}_forest_lai_500m.tif  (CONUS extent)
Step 2 (step2_pft_upscale.py) reads these to produce the 144×96 ELM/CLM input.
"""

import os
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.transform import Affine
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
LAI_DIR   = "/mnt/cephfs-mount/hangkai/CONUS_LAI/backup_all"
DEPTH_DIR = "/mnt/cephfs-mount/hangkai/CONUS_Forest_Depth"
PFT_DIR   = "/mnt/cephfs-mount/hangkai/PFT"
OUT_DIR   = "/mnt/cephfs-mount/hangkai/LAI_500m"

YEARS   = range(2001, 2015)
MONTHS  = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

FILL_VALUE = -9999.0
CHUNK_ROWS = 500   # 30m rows read at once; raise if RAM allows

# Band layout in output GeoTIFF (1-indexed for rasterio)
BAND_INTERIOR_MEAN  = 1
BAND_EXTERIOR_MEAN  = 2
BAND_WEIGHTED_MEAN  = 3
BAND_INTERIOR_COUNT = 4
BAND_EXTERIOR_COUNT = 5
BAND_TOTAL_COUNT    = 6

# ── Module-level caches ────────────────────────────────────────────────────────
_map_cache = {}   # (depth dims+bounds, pft dims+bounds) → (px_col, py_row, conus_in_pft)


# ── 30m → 500m index mapping, CONUS-clipped ───────────────────────────────────
def compute_mapping(depth_bounds, depth_crs, depth_w, depth_h,
                    pft_bounds, pft_w, pft_h):
    """
    Return:
        px_col      : int32 (depth_w,) — PFT column for each 30m column
        py_row      : int32 (depth_h,) — PFT row    for each 30m row
        conus_in_pft: (x0, y0, cw, ch) — CONUS rectangle in PFT pixel space

    The PFT file may be global.  conus_in_pft identifies which sub-rectangle
    of the global PFT grid the CONUS LAI data actually touches, so all
    downstream arrays are sized to CONUS only, not the full globe.
    """
    west, south, east, north = transform_bounds(depth_crs, "EPSG:4326",
                                                *depth_bounds)
    lon = np.linspace(west,  east,  depth_w)
    lat = np.linspace(north, south, depth_h)   # north → south

    span_lon = pft_bounds.right - pft_bounds.left
    span_lat = pft_bounds.top   - pft_bounds.bottom

    px = np.floor((lon - pft_bounds.left) / span_lon * pft_w).astype(np.int32)
    py = np.floor((pft_bounds.top  - lat) / span_lat * pft_h).astype(np.int32)

    np.clip(px, 0, pft_w - 1, out=px)
    np.clip(py, 0, pft_h - 1, out=py)

    # CONUS bounding box in PFT pixel space
    x0, x1 = int(px.min()), int(px.max())
    y0, y1 = int(py.min()), int(py.max())
    conus_in_pft = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)   # (x0, y0, cw, ch)

    return px, py, conus_in_pft


# ── Core aggregation for one month (CONUS-local PFT space) ────────────────────
def aggregate_month(lai_file, depth_arr, px_col, py_row, H, W, conus_in_pft):
    """
    Read 30m LAI in CHUNK_ROWS-row windows and bin pixels into 500m PFT cells,
    working entirely in CONUS-local PFT coordinates so array sizes are
    proportional to CONUS (not the full global PFT grid).

    Returns
    -------
    lai_means    : float32 (3, ch, cw)  — FILL_VALUE where no valid pixels
    pixel_counts : float32 (3, ch, cw)  — 0 where no valid pixels
        [0] interior (depth==5)  [1] exterior (depth 1-4)  [2] weighted (1-5)
    """
    pft_x0, pft_y0, pft_cw, pft_ch = conus_in_pft
    n_cells = pft_cw * pft_ch   # CONUS cells only, not the full globe

    sums   = [np.zeros(n_cells, np.float64) for _ in range(3)]
    counts = [np.zeros(n_cells, np.int64)   for _ in range(3)]

    with rasterio.open(lai_file) as lai_src:
        for r0 in range(0, H, CHUNK_ROWS):
            r1    = min(r0 + CHUNK_ROWS, H)
            win   = Window(0, r0, W, r1 - r0)

            lai_c = lai_src.read(1, window=win).astype(np.float32)
            dep_c = depth_arr[r0:r1]               # uint8 (chunk_h, W)
            py_c  = py_row[r0:r1]                  # int32 (chunk_h,)

            lai_ok = (lai_c > 0) & (lai_c < 100) & ~np.isnan(lai_c)

            cat_masks = [
                lai_ok & (dep_c == 5),                        # interior
                lai_ok & (dep_c >= 1) & (dep_c <= 4),         # exterior
                lai_ok & (dep_c >= 1) & (dep_c <= 5),         # weighted
            ]

            for cat, mask in enumerate(cat_masks):
                rr, cc = np.where(mask)
                if rr.size == 0:
                    continue

                # CONUS-local indices in the PFT sub-grid
                px_local = px_col[cc].astype(np.int32) - pft_x0   # 0 … cw-1
                py_local = py_c  [rr].astype(np.int32) - pft_y0   # 0 … ch-1
                lin      = py_local * pft_cw + px_local             # 0 … n_cells-1

                vals = lai_c[rr, cc].astype(np.float64)
                sums  [cat] += np.bincount(lin, weights=vals, minlength=n_cells)
                counts[cat] += np.bincount(lin,               minlength=n_cells)

    lai_means    = np.full((3, pft_ch, pft_cw), FILL_VALUE, np.float32)
    pixel_counts = np.zeros((3, pft_ch, pft_cw), np.float32)

    for cat in range(3):
        s  = sums  [cat].reshape(pft_ch, pft_cw)
        c  = counts[cat].reshape(pft_ch, pft_cw)
        ok = c > 0
        lai_means   [cat][ok] = (s[ok] / c[ok]).astype(np.float32)
        pixel_counts[cat]     = c.astype(np.float32)

    return lai_means, pixel_counts


# ── Year-level processing ──────────────────────────────────────────────────────
def process_year(year, outer_pbar=None):

    def log(msg):
        if outer_pbar: outer_pbar.write(msg)
        else: print(msg, flush=True)

    log(f"\n{'='*64}")
    log(f"  STEP 1 — YEAR {year}")
    log(f"{'='*64}")
    t_year = datetime.now()

    # 1. Load Forest Depth (30m Albers — reference grid)
    depth_file = f"{DEPTH_DIR}/LCMAP_CU_{year}_V13_LCPRI.tif"
    log(f"  Loading depth:  {depth_file}")
    with rasterio.open(depth_file) as src:
        depth_data    = src.read(1).astype(np.uint8)
        depth_bounds  = src.bounds
        depth_crs     = src.crs
        depth_profile = src.profile
    H, W = depth_profile["height"], depth_profile["width"]
    log(f"  Depth grid: {W} × {H}  ({W * H / 1e6:.1f} M pixels)")

    # 2. Read PFT file metadata only — may be global, do NOT load pixel data here
    pft_file = f"{PFT_DIR}/ELM_PFT_{year}-WGS84-merged.tif"
    log(f"  PFT metadata:   {pft_file}")
    with rasterio.open(pft_file) as src:
        pft_w, pft_h  = src.width, src.height
        pft_bounds    = src.bounds
        pft_transform = src.transform
        pft_crs       = src.crs
    log(f"  PFT grid (full): {pft_w} × {pft_h}  ({pft_w * pft_h / 1e6:.1f} M cells)")

    # 3. Compute 30m → 500m mapping + CONUS footprint in PFT space (cached)
    map_key = (W, H, str(depth_bounds), pft_w, pft_h, str(pft_bounds))
    if map_key not in _map_cache:
        log("  Computing 30m → 500m index mapping …")
        px_col, py_row, conus_in_pft = compute_mapping(
            depth_bounds, depth_crs, W, H, pft_bounds, pft_w, pft_h)
        _map_cache[map_key] = (px_col, py_row, conus_in_pft)
        x0, y0, cw, ch = conus_in_pft
        log(f"  CONUS in PFT grid: cols {x0}–{x0+cw-1}, rows {y0}–{y0+ch-1}"
            f"  ({cw}×{ch} = {cw*ch/1e6:.2f} M cells  vs {pft_w*pft_h/1e6:.1f} M global)")
    px_col, py_row, conus_in_pft = _map_cache[map_key]
    pft_x0, pft_y0, pft_cw, pft_ch = conus_in_pft

    # 4. Geographic transform for the CONUS sub-region of the PFT grid
    #    pft_transform maps (col, row) → (lon, lat):
    #      lon = pft_transform.c + col * pft_transform.a
    #      lat = pft_transform.f + row * pft_transform.e  (e < 0 for north-up)
    conus_transform = Affine(
        pft_transform.a, 0.0,
        pft_transform.c + pft_x0 * pft_transform.a,   # left-edge longitude
        0.0, pft_transform.e,
        pft_transform.f + pft_y0 * pft_transform.e,   # top-edge latitude
    )

    # 5. Output profile — CONUS extent only, NOT the full PFT grid
    out_dir = Path(OUT_DIR) / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_profile = {
        "driver"   : "GTiff",
        "dtype"    : "float32",
        "width"    : pft_cw,
        "height"   : pft_ch,
        "count"    : 6,
        "crs"      : pft_crs,
        "transform": conus_transform,
        "compress" : "lzw",
        "nodata"   : FILL_VALUE,
    }

    # 6. Process each month
    pbar_m = tqdm(
        enumerate(MONTHS), total=12,
        desc=f"  {year}", unit="month", leave=False, dynamic_ncols=True,
    )
    missing = 0
    for m_idx, month in pbar_m:
        lai_file = f"{LAI_DIR}/{year}_{month}_LAI.tif"
        out_file = out_dir / f"{year}_{month}_forest_lai_500m.tif"

        if out_file.exists():
            pbar_m.write(f"    Skipping {month} — already done")
            continue

        if not os.path.exists(lai_file):
            missing += 1
            pbar_m.write(f"    WARNING: missing {lai_file}")
            continue

        lai_means, pixel_counts = aggregate_month(
            lai_file, depth_data, px_col, py_row, H, W, conus_in_pft)
        with rasterio.open(out_file, "w", **out_profile) as dst:
            dst.write(lai_means   [0], BAND_INTERIOR_MEAN)
            dst.write(lai_means   [1], BAND_EXTERIOR_MEAN)
            dst.write(lai_means   [2], BAND_WEIGHTED_MEAN)
            dst.write(pixel_counts[0], BAND_INTERIOR_COUNT)
            dst.write(pixel_counts[1], BAND_EXTERIOR_COUNT)
            dst.write(pixel_counts[2], BAND_TOTAL_COUNT)

    pbar_m.close()
    if missing:
        log(f"  {missing} month(s) missing for {year}")

    log(f"  Year {year} done in {datetime.now() - t_year}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("STEP 1: 30m LAI + Depth  →  500m WGS84 Forest LAI (CONUS-clipped)")
    print("=" * 64)
    print(f"Years:      {YEARS.start} – {YEARS.stop - 1}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Bands:      1-3 = LAI means (interior / exterior / weighted)")
    print(f"            4-6 = pixel counts (interior / exterior / total forest)")
    print(f"Chunk rows: {CHUNK_ROWS}")
    print(f"Note:       PFT may be global; output is clipped to CONUS extent.")
    print()

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    t_total = datetime.now()

    with tqdm(list(YEARS), desc="Years", unit="yr", dynamic_ncols=True) as outer:
        for year in outer:
            outer.set_postfix(year=year)
            try:
                process_year(year, outer)
            except Exception as e:
                outer.write(f"ERROR year {year}: {e}")
                import traceback; traceback.print_exc()

    print(f"\nDone!  Total time: {datetime.now() - t_total}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
