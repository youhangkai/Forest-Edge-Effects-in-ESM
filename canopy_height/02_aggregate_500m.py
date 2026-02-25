#!/usr/bin/env python3
"""
step1_canopy_height.py

Aggregates GLAD 2020 Forest Height tiles (0.00025° WGS84) to the 500m WGS84
PFT grid over CONUS, classified by forest interior / exterior using the 30m
Albers forest-depth raster.

Tile naming convention:  2020_{N}N_{W}W.tif
  {N}N = NORTHERN (top) edge of tile, extends 10° SOUTH
  {W}W = WESTERN  edge of tile, extends 10° EAST
  e.g. 2020_40N_090W → lat 30–40°N, lon 90–80°W

Processing (per 500-row Albers chunk):
  1. Transform chunk's Albers bounds → WGS84
  2. Mosaic all canopy-height tiles overlapping that WGS84 extent
  3. Reproject the mosaic from WGS84 → same Albers grid as the depth chunk
  4. Mask valid pixels: resampled height > 0  AND  forest depth ∈ [1–5]
  5. Accumulate Σ(height) and Σ(count) by [category, CONUS-500m-cell]
     Category 0 — all forest  (depth 1–5)
     Category 1 — exterior    (depth 1–4)
     Category 2 — interior    (depth == 5)

Output (3 files, one per scenario):
  Canopy_Height/processed/canopy_height_500m_v1_all_forests.tif
  Canopy_Height/processed/canopy_height_500m_v2_exterior_forests.tif
  Canopy_Height/processed/canopy_height_500m_v3_interior_forests.tif

Each file: float32, 1 band, CONUS extent, 500m WGS84, FILL_VALUE=-9999 where
no valid pixels.  (PFT stratification can be applied in a subsequent step by
masking with ELM_PFT_2020-WGS84-merged.tif.)
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine
from rasterio.windows import Window
from pathlib import Path
from datetime import datetime
from glob import glob
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
CH_DIR     = "/mnt/cephfs-mount/hangkai/Canopy_Height/raw"
DEPTH_FILE = "/mnt/cephfs-mount/hangkai/CONUS_Forest_Depth/LCMAP_CU_2020_V13_LCPRI.tif"
PFT_FILE   = "/mnt/cephfs-mount/hangkai/PFT/ELM_PFT_2020-WGS84-merged.tif"
OUT_DIR    = "/mnt/cephfs-mount/hangkai/Canopy_Height/processed"

FILL_VALUE = np.float32(-9999.0)
CH_NODATA  = 0        # Potapov 2020: 0 = non-forest / no data; valid range 1–65 m
CHUNK_ROWS = 500      # depth rows per chunk (500 × 30 m = 15 km)

# Scenarios: (output_suffix, category_index)
#   category 0 = all forest (depth 1–5)
#   category 1 = exterior   (depth 1–4)
#   category 2 = interior   (depth == 5)
SCENARIOS = [
    ("v1_all_forests",      0),
    ("v2_exterior_forests", 1),
    ("v3_interior_forests", 2),
]


# ── 30m → 500m index mapping ───────────────────────────────────────────────────
def compute_mapping(depth_bounds, depth_crs, depth_w, depth_h,
                    pft_bounds, pft_w, pft_h):
    """
    Map each 30m Albers pixel to its 500m WGS84 PFT cell.
    Returns px_col (int32, depth_w), py_row (int32, depth_h),
    and conus_in_pft = (x0, y0, cw, ch).
    """
    west, south, east, north = transform_bounds(
        depth_crs, "EPSG:4326", *depth_bounds)

    lon = np.linspace(west,  east,  depth_w)
    lat = np.linspace(north, south, depth_h)   # N → S

    span_lon = pft_bounds.right - pft_bounds.left
    span_lat = pft_bounds.top   - pft_bounds.bottom

    px = np.floor((lon - pft_bounds.left) / span_lon * pft_w).astype(np.int32)
    py = np.floor((pft_bounds.top  - lat) / span_lat * pft_h).astype(np.int32)

    np.clip(px, 0, pft_w - 1, out=px)
    np.clip(py, 0, pft_h - 1, out=py)

    x0, x1 = int(px.min()), int(px.max())
    y0, y1 = int(py.min()), int(py.max())
    conus_in_pft = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    return px, py, conus_in_pft


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    print("=" * 64)
    print("STEP 1 — Canopy Height 500m Aggregation (CONUS 2020)")
    print("=" * 64)

    # 1. Load forest depth (30m Albers, full CONUS)
    print(f"\nLoading depth: {DEPTH_FILE}")
    with rasterio.open(DEPTH_FILE) as src:
        depth_data   = src.read(1).astype(np.uint8)
        depth_bounds = src.bounds
        depth_crs    = src.crs
        depth_tf     = src.transform
    H, W = depth_data.shape
    print(f"  Depth grid: {W} × {H}  ({W * H / 1e6:.1f} M pixels)")

    # 2. PFT metadata (windowed read later)
    with rasterio.open(PFT_FILE) as src:
        pft_w, pft_h = src.width, src.height
        pft_bounds   = src.bounds
        pft_tf       = src.transform

    # 3. 30m → 500m mapping + CONUS bounding box in PFT space
    print("Computing 30m → 500m index mapping …")
    px_col, py_row, conus_in_pft = compute_mapping(
        depth_bounds, depth_crs, W, H, pft_bounds, pft_w, pft_h)
    pft_x0, pft_y0, pft_cw, pft_ch = conus_in_pft
    n_cells = pft_cw * pft_ch
    print(f"  CONUS in PFT grid: cols {pft_x0}–{pft_x0+pft_cw-1}, "
          f"rows {pft_y0}–{pft_y0+pft_ch-1}  "
          f"({pft_cw}×{pft_ch} = {n_cells/1e6:.2f} M cells)")

    # 4. Open all canopy height tiles (keep open for efficient windowed reads)
    ch_files = sorted(glob(f"{CH_DIR}/2020_*.tif"))
    print(f"\nCanopy height tiles: {len(ch_files)}")
    for f in ch_files:
        with rasterio.open(f) as src:
            print(f"  {Path(f).name}: lat {src.bounds.bottom:.0f}–{src.bounds.top:.0f}°N, "
                  f"lon {src.bounds.left:.0f}–{src.bounds.right:.0f}°E")
    ch_srcs = [rasterio.open(f) for f in ch_files]

    # 5. Accumulation arrays: [3 categories] × [n_cells]
    #    sums[cat][cell]   = Σ height values
    #    counts[cat][cell] = number of valid pixels
    sums   = [np.zeros(n_cells, np.float64) for _ in range(3)]
    counts = [np.zeros(n_cells, np.int64)   for _ in range(3)]

    # 6. Process depth in chunks
    print(f"\nProcessing {H} rows in {CHUNK_ROWS}-row chunks …")
    n_chunks = (H + CHUNK_ROWS - 1) // CHUNK_ROWS
    pbar = tqdm(range(0, H, CHUNK_ROWS), total=n_chunks,
                unit="chunk", dynamic_ncols=True)

    for r0 in pbar:
        r1 = min(r0 + CHUNK_ROWS, H)
        h  = r1 - r0

        dep_c = depth_data[r0:r1]   # (h, W) uint8
        py_c  = py_row[r0:r1]       # (h,)   int32

        # ── a. WGS84 bounds of this Albers chunk ──────────────────────────
        #  depth_tf: top-left corner + pixel size
        #  e < 0 (north-up), so bottom = f + r1*e, top = f + r0*e
        chunk_albers = (
            depth_tf.c,                           # left
            depth_tf.f + r1 * depth_tf.e,         # bottom  (e < 0)
            depth_tf.c + W  * depth_tf.a,         # right
            depth_tf.f + r0 * depth_tf.e,         # top
        )
        chunk_wgs84 = transform_bounds(depth_crs, "EPSG:4326", *chunk_albers)

        # ── b. Mosaic overlapping canopy height tiles ──────────────────────
        try:
            ch_merged, ch_tf_m = rio_merge(
                ch_srcs,
                bounds=chunk_wgs84,
                nodata=float(CH_NODATA),
            )
            ch_merged = ch_merged[0].astype(np.float32)   # (rows_wgs, cols_wgs)
        except Exception:
            continue   # no tile overlaps this chunk

        if ch_merged.max() <= 0:
            continue   # all nodata

        # ── c. Reproject merged WGS84 → Albers at 30 m for this chunk ─────
        chunk_tf = Affine(
            depth_tf.a, 0.0, depth_tf.c,
            0.0, depth_tf.e, depth_tf.f + r0 * depth_tf.e,
        )
        ch_albers = np.zeros((h, W), dtype=np.float32)
        reproject(
            source=ch_merged,
            destination=ch_albers,
            src_transform=ch_tf_m,
            src_crs="EPSG:4326",
            dst_transform=chunk_tf,
            dst_crs=depth_crs,
            resampling=Resampling.bilinear,
            src_nodata=float(CH_NODATA),
            dst_nodata=0.0,
        )

        # ── d. Valid pixels: height > 0 AND depth ∈ [1–5] ─────────────────
        forest = (dep_c >= 1) & (dep_c <= 5)
        ch_ok  = (ch_albers > 0) & (ch_albers < 200)   # 200 m upper bound

        cat_masks = [
            ch_ok & forest,                                   # all  (depth 1–5)
            ch_ok & (dep_c >= 1) & (dep_c <= 4),             # exterior
            ch_ok & (dep_c == 5),                             # interior
        ]

        # ── e. Bin into CONUS 500m cells ──────────────────────────────────
        for cat, mask in enumerate(cat_masks):
            rr, cc = np.where(mask)
            if rr.size == 0:
                continue

            px_local = px_col[cc].astype(np.int32) - pft_x0   # 0…cw-1
            py_local = py_c  [rr].astype(np.int32) - pft_y0   # 0…ch-1

            # Safety clip (shouldn't be needed but guard against edge rounding)
            valid = (
                (px_local >= 0) & (px_local < pft_cw) &
                (py_local >= 0) & (py_local < pft_ch)
            )
            if not valid.all():
                px_local = px_local[valid]
                py_local = py_local[valid]
                rr       = rr[valid]
                cc       = cc[valid]

            lin  = py_local * pft_cw + px_local
            vals = ch_albers[rr, cc].astype(np.float64)

            sums  [cat] += np.bincount(lin, weights=vals, minlength=n_cells)
            counts[cat] += np.bincount(lin,               minlength=n_cells).astype(np.int64)

        pbar.set_postfix(rows=f"{r0}–{r1}")

    pbar.close()
    for s in ch_srcs:
        s.close()

    print(f"\nAggregation done in {datetime.now() - t0}")

    # 7. Compute means and write output GeoTIFFs
    conus_tf = Affine(
        pft_tf.a, 0.0, pft_tf.c + pft_x0 * pft_tf.a,
        0.0, pft_tf.e, pft_tf.f + pft_y0 * pft_tf.e,
    )
    out_profile = {
        "driver"   : "GTiff",
        "dtype"    : "float32",
        "width"    : pft_cw,
        "height"   : pft_ch,
        "count"    : 1,
        "crs"      : "EPSG:4326",
        "transform": conus_tf,
        "compress" : "lzw",
        "nodata"   : float(FILL_VALUE),
    }

    print("\nWriting output files …")
    for suffix, cat in SCENARIOS:
        out_path = Path(OUT_DIR) / f"canopy_height_500m_{suffix}.tif"

        s   = sums[cat]  .reshape(pft_ch, pft_cw)
        c   = counts[cat].reshape(pft_ch, pft_cw)
        ok  = c > 0
        out = np.full((pft_ch, pft_cw), FILL_VALUE, np.float32)
        out[ok] = (s[ok] / c[ok]).astype(np.float32)

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out, 1)

        valid_n = ok.sum()
        mean_h  = out[ok].mean() if valid_n > 0 else float('nan')
        print(f"  {out_path.name}: {valid_n:,} valid cells, "
              f"mean height = {mean_h:.2f} m")

    print(f"\nDone!  Total time: {datetime.now() - t0}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
