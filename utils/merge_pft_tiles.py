#!/usr/bin/env python3
"""
Script to merge TIFF files by year.
For each year, merges the two spatial tiles into a single TIFF file.
"""

import os
import glob
from pathlib import Path
from rasterio.merge import merge
import rasterio

def merge_tifs_by_year(input_dir, output_dir=None):
    """
    Merge TIFF files by year in the specified directory.

    Args:
        input_dir: Directory containing the TIFF files
        output_dir: Directory for output files (defaults to input_dir)
    """
    if output_dir is None:
        output_dir = input_dir

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find all unique years
    tif_files = list(input_path.glob("ELM_PFT_*-WGS84-*.tif"))
    years = sorted(set(f.name.split('_')[2].split('-')[0] for f in tif_files))

    print(f"Found {len(years)} years to process: {years[0]} to {years[-1]}")
    print()

    for year in years:
        # Find all files for this year
        pattern = f"ELM_PFT_{year}-WGS84-*.tif"
        year_files = sorted(input_path.glob(pattern))

        if len(year_files) == 0:
            print(f"⚠️  No files found for year {year}")
            continue

        print(f"Processing {year}: {len(year_files)} files")
        for f in year_files:
            print(f"  - {f.name}")

        # Open all files for this year
        src_files_to_mosaic = []
        try:
            for fp in year_files:
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)

            # Merge the files
            mosaic, out_trans = merge(src_files_to_mosaic)

            # Copy metadata from the first file
            out_meta = src_files_to_mosaic[0].meta.copy()

            # Update metadata with new dimensions and transform
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw"  # Use LZW compression to reduce file size
            })

            # Write the merged file
            output_file = output_path / f"ELM_PFT_{year}-WGS84-merged.tif"
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)

            print(f"✓ Created: {output_file.name}")
            print(f"  Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
            print()

        finally:
            # Close all source files
            for src in src_files_to_mosaic:
                src.close()

if __name__ == "__main__":
    input_directory = "/mnt/cephfs-mount/hangkai/PFT"
    print(f"Merging TIFF files in: {input_directory}")
    print("=" * 60)
    merge_tifs_by_year(input_directory)
    print("=" * 60)
    print("All done!")
