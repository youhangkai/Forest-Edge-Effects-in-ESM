# Forest Edge Effects via E3SM

Preprocessing pipeline for studying forest edge effects using the E3SM Land Model (ELM).
Generates spatially explicit canopy height and LAI input files for CONUS forests,
stratified by interior vs. exterior forest pixels using forest depth data.

## Scientific Context

Forest edges differ structurally and functionally from forest interiors.
This project creates a 3×3 scenario matrix of ELM surface/stream inputs:

|              | CH only          | LAI only          | Both             |
|--------------|------------------|-------------------|------------------|
| **area-wtd** | CH_v1 + LAI_orig | CH_orig + LAI_v1  | CH_v1 + LAI_v1   |
| **interior** | CH_v3 + LAI_orig | CH_orig + LAI_v3  | CH_v3 + LAI_v3   |
| **exterior** | CH_v2 + LAI_orig | CH_orig + LAI_v2  | CH_v2 + LAI_v2   |

---

## Repository Structure

```
forest-edge-effects-via-e3sm/
├── canopy_height/
│   ├── 01_download_tiles.sh      # Download GLAD 2020 Forest Height tiles (CONUS)
│   ├── 02_aggregate_500m.py      # 0.00025° WGS84 → 500m WGS84, 3 scenarios
│   └── 03_update_surfdata.py     # 500m → 0.5° per PFT; replace MONTHLY_HEIGHT_TOP
├── lai/
│   ├── 01_aggregate_500m.py      # 30m Albers LAI + depth → 500m WGS84 (2001–2014)
│   └── 02_update_lai_nc.py       # 500m → 0.5° per PFT; replace MODISPFTLAI NC
└── utils/
    └── merge_pft_tiles.py        # Merge multi-tile PFT GeoTIFFs by year
```

---

## Input Data

| Dataset | Source | Resolution | Notes |
|---------|--------|------------|-------|
| GLAD Forest Height 2020 | [UMD GLAD](https://glad.umd.edu/) | 0.00025° WGS84 | Potapov et al. 2020; 0 = no-data |
| LCMAP Forest Depth | Kang et al. (in prep.) | 30m Albers | `LCMAP_CU_{year}_V13_LCPRI.tif`; values 1–5 (5=interior); derived from USGS LCMAP |
| MODIS PFT (500m WGS84) | [build_surface_dataset_for_ELM](https://github.com/daleihao/build_surface_dataset_for_ELM) | 500m WGS84 | `ELM_PFT_{year}-WGS84-merged.tif`; PFTs 1–8 are trees |
| ELM surfdata template | ELM ancillary | 0.5° global | `surfdata_0.5x0.5_simyr2010_c251122.nc` |
| MODIS PFT LAI stream | ELM ancillary | 0.5° global | `MODISPFTLAI_0.5x0.5_c140711.nc` |
| CONUS LAI (30m) | [Landsat-LAI_YanghuiKang](https://github.com/youhangkai/Landsat-LAI_YanghuiKang) | 30m Albers | Monthly, 2001–2014; derived from Landsat using the model in Kang et al. |

---

## Canopy Height Pipeline

### Step 0 — Download raw tiles
```bash
bash canopy_height/01_download_tiles.sh
```
Downloads 25 GLAD 2020 Forest Height tiles covering CONUS into
`Canopy_Height/raw/`. Tiles are 10°×10°, named by their **northern** edge
(e.g., `2020_50N_120W` covers lat 40–50°N, lon 120–110°W).

### Step 1 — Aggregate to 500m
```bash
python canopy_height/02_aggregate_500m.py
```
Per 500-row chunk of the 30m Albers forest depth grid:
1. Computes WGS84 bounding box of the chunk
2. Mosaics overlapping GLAD tiles with `rasterio.merge`
3. Reprojects mosaic to 30m Albers via bilinear resampling
4. Masks valid pixels: height > 0 AND forest depth ∈ [1–5]
5. Bins into 500m WGS84 PFT cells with `np.bincount`

Outputs (`Canopy_Height/processed/`):

| File | Scenario | Pixels used |
|------|----------|-------------|
| `canopy_height_500m_v1_all_forests.tif` | Area-weighted | depth 1–5 |
| `canopy_height_500m_v2_exterior_forests.tif` | Exterior only | depth 1–4 |
| `canopy_height_500m_v3_interior_forests.tif` | Interior only | depth == 5 |

### Step 2 — Replace surfdata MONTHLY_HEIGHT_TOP
```bash
python canopy_height/03_update_surfdata.py
```
Aggregates 500m CH to the 0.5° ELM grid per forest PFT (1–8) using
count-weighted binning. Replaces `MONTHLY_HEIGHT_TOP` (shape `12×17×360×720`)
for tree PFTs at CONUS cells. Non-CONUS cells and non-tree PFTs are unchanged.

Outputs (`surfdata_revised/`):

| File | Scenario |
|------|----------|
| `surfdata_0.5x0.5_CH_v1_all_forests.nc` | Area-weighted |
| `surfdata_0.5x0.5_CH_v2_exterior_forests.nc` | Exterior only |
| `surfdata_0.5x0.5_CH_v3_interior_forests.nc` | Interior only |

---

## LAI Pipeline

### Step 1 — Aggregate to 500m
```bash
python lai/01_aggregate_500m.py
```
For each month (2001–2014), reads 30m Albers LAI and forest depth in
500-row chunks, bins valid forest pixels into 500m WGS84 cells. Outputs
6-band GeoTIFFs (interior mean / exterior mean / weighted mean / 3 counts).

Outputs (`LAI_500m/{year}/{year}_{month}_forest_lai_500m.tif`):
- Band 1: Interior LAI mean (depth == 5)
- Band 2: Exterior LAI mean (depth 1–4)
- Band 3: Area-weighted LAI mean (depth 1–5)
- Bands 4–6: Pixel counts per category

### Step 2 — Replace MODISPFTLAI NC
```bash
python lai/02_update_lai_nc.py
```
Aggregates 500m LAI to the 0.5° NC grid per forest PFT (1–8) using
count-weighted averaging. Replaces forest LAI in the MODISPFTLAI NC file
for 156 months (Jan 2001–Dec 2013).

Outputs (`LAI_revised/`):

| File | Scenario |
|------|----------|
| `MODISPFTLAI_0.5x0.5_v1_all_forests.nc` | Area-weighted |
| `MODISPFTLAI_0.5x0.5_v2_exterior_forests.nc` | Exterior only |
| `MODISPFTLAI_0.5x0.5_v3_interior_forests.nc` | Interior only |

---

## Utility

### Merge PFT tiles
```bash
python utils/merge_pft_tiles.py
```
Merges multi-tile PFT GeoTIFFs (e.g., two spatial tiles from ELM ancillary)
into a single merged file per year. Run once before the main pipelines if
the PFT data was downloaded as separate tiles.

---

## Dependencies

```
python >= 3.10
numpy
rasterio
scipy
tqdm
```

Install with:
```bash
pip install numpy rasterio scipy tqdm
```

---

## Expected Run Times (on ~256 GB RAM server)

| Script | Time |
|--------|------|
| `canopy_height/02_aggregate_500m.py` | ~1h 45min |
| `canopy_height/03_update_surfdata.py` | ~1 min |
| `lai/01_aggregate_500m.py` | ~several hours (14 years × 12 months) |
| `lai/02_update_lai_nc.py` | ~minutes |

---

## File Path Configuration

All input/output paths are defined as constants at the top of each script.
Edit these before running:

```python
# Example from canopy_height/02_aggregate_500m.py
CH_DIR     = "/path/to/Canopy_Height/raw"
DEPTH_FILE = "/path/to/CONUS_Forest_Depth/LCMAP_CU_2020_V13_LCPRI.tif"
PFT_FILE   = "/path/to/PFT/ELM_PFT_2020-WGS84-merged.tif"
OUT_DIR    = "/path/to/Canopy_Height/processed"
```

---

## Citation

30m Landsat LAI for CONUS:
The 30m monthly LAI data over CONUS is derived from Landsat imagery using the model described in:
Kang, Y., et al. — see [Landsat-LAI_YanghuiKang](https://github.com/youhangkai/Landsat-LAI_YanghuiKang) for the full model code and methodology.

Forest canopy height data:
The canopy height data used in this project is the **Forest Height 2020** layer from the
[GLAD Global Land Cover and Land Use Change 2000–2020](https://glad.umd.edu/dataset/GLCLUC2020) dataset
(GLAD lab, University of Maryland). Forest is defined as wildland, managed, and planted tree cover
(including agroforestry and orchards) with height >= 3m. Forest height was mapped globally using a
Landsat-based model calibrated with GEDI observations. Data are provided as 10x10 degree GeoTIFF tiles
(WGS84, pixel value = forest height in meters).
Potapov, P., Li, X., Hernandez-Serna, A., Tyukavina, A., Hansen, M.C., Kommareddy, A., Pickens, A.,
Turubanova, S., Tang, H., Silva, C.E., Armston, J., Dubayah, R., Blair, J.B., Hofton, M. (2020).
Mapping and monitoring global forest canopy height through integration of GEDI and Landsat data.
*Remote Sensing of Environment*, 112165. https://doi.org/10.1016/j.rse.2020.112165

Forest depth data:
The forest depth dataset (values 1–5, where 5 = interior forest) was derived from USGS LCMAP
land cover data by the authors of this project. Please cite:
Kang, Y., et al. (in preparation). Human-induced disturbances accelerate forest edge expansion in the U.S.

MODIS PFT (500m WGS84):
The ELM PFT dataset was generated following the pipeline described in:
https://github.com/daleihao/build_surface_dataset_for_ELM
