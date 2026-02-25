#!/bin/bash
# Download GLAD Forest Height 2020 tiles covering CONUS
# Tiles are 10°×10°, named by SW corner, extending north and east.
# Source: https://gladxfer.umd.edu/users/Potapov/GLCLUC2020/Forest_height_2020/

BASE="https://gladxfer.umd.edu/users/Potapov/GLCLUC2020/Forest_height_2020"
OUT_DIR="/mnt/cephfs-mount/hangkai/Canopy_Height/raw"

# CONUS tiles confirmed to exist on server:
#   Row 20N (lat 20–30°N): southern Florida + Texas
#   Row 30N (lat 30–40°N): southern half of CONUS
#   Row 40N (lat 40–50°N): northern half of CONUS, includes PNW coast
declare -a TILES=(
    "2020_20N_060W"
    "2020_20N_070W"
    "2020_20N_080W"
    "2020_20N_090W"
    "2020_20N_100W"
    "2020_20N_110W"
    "2020_30N_080W"
    "2020_30N_090W"
    "2020_30N_100W"
    "2020_30N_110W"
    "2020_30N_120W"
    "2020_40N_080W"
    "2020_40N_090W"
    "2020_40N_100W"
    "2020_40N_110W"
    "2020_40N_120W"
    "2020_40N_130W"
)

echo "====================================================="
echo "Downloading GLAD Forest Height 2020 — CONUS tiles"
echo "Output: ${OUT_DIR}"
echo "Total tiles: ${#TILES[@]}"
echo "====================================================="
echo ""

n_ok=0
n_skip=0
n_fail=0

for tile in "${TILES[@]}"; do
    fname="${tile}.tif"
    out_file="${OUT_DIR}/${fname}"

    if [ -f "${out_file}" ]; then
        size=$(du -h "${out_file}" | cut -f1)
        echo "[SKIP] ${fname} already exists (${size})"
        ((n_skip++))
        continue
    fi

    echo "[DOWN] ${fname} ..."
    wget -c -q --show-progress \
         --timeout=300 --tries=5 --waitretry=30 \
         -O "${out_file}" \
         "${BASE}/${fname}"

    if [ $? -eq 0 ]; then
        size=$(du -h "${out_file}" | cut -f1)
        echo "[OK]   ${fname} → ${size}"
        ((n_ok++))
    else
        echo "[FAIL] ${fname}"
        rm -f "${out_file}"   # remove incomplete file
        ((n_fail++))
    fi
done

echo ""
echo "====================================================="
echo "Done: ${n_ok} downloaded, ${n_skip} skipped, ${n_fail} failed"
echo "====================================================="
ls -lh "${OUT_DIR}/"
