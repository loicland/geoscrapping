import pandas as pd
import numpy as np
import os

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Main Execution
DATA_FOLDER = '/home/ign.fr/llandrieu/Documents/data/plonk/mapillary_full'
DATA_FOLDER = '/var/data/llandrieu/mapillary_full/'
OUT_FOLDER = '/var/data/llandrieu/geoscrapping/'
cell_size  = 0.1
min_img_per_cell = 2
compute_size = False

def lat_lon_to_grid_index(lat, lon, cell_size=10):
    lat_idx = (lat + 90) // (0.009 * cell_size) # 0.009° is approx 1km
    lon_idx = (lon + 180) // (0.009 * cell_size / np.cos(np.radians(lat)))
    return lat_idx.astype('int'), lon_idx.astype('int')

# Set up Dask DataFrame
ordered_ddf = dd.read_csv(os.path.join(DATA_FOLDER, "*.csv"), assume_missing=True)

# Shuffle the Dask DataFrame, avoid 
ddf = ordered_ddf.sample(frac=1, random_state=42)

# This will give you a tuple of (Delayed, int)

if compute_size:
    print("Computing size of dataframe...")
    shape = ddf.shape
    with ProgressBar():
        num_rows = shape[0].compute()
    print(f"Number of rows: {num_rows:,}")

# Convert lat/lon to grid indices
ddf["lat_idx"], ddf["lon_idx"] = lat_lon_to_grid_index(ddf["latitude"], ddf["longitude"], cell_size)

# Compute resulting dataframe and get unique grid cells

print(f"Pruning on a {cell_size}km grid...")
with ProgressBar():
    # Group by lat_idx and lon_idx and aggregate to get one row for each group
    aggregated_ddf = ddf.groupby(["lat_idx", "lon_idx"]).agg({
        "image_id": ["first", "count"],
        "latitude": "first",
        "longitude": "first"
    }).reset_index()

    # Rename columns for clarity
    aggregated_ddf.columns = ["lat_idx", "lon_idx", "image_id", "image_count", "latitude", "longitude"]

    # Filter out cells with less than N images
    aggregated_ddf = aggregated_ddf[aggregated_ddf["image_count"] >= min_img_per_cell].compute()


print(f"Selected {aggregated_ddf.shape[0]:,} cells ")

# Merge the unique_cells with the original DDF to get the full rows of the selected unique cells
 
print("Saving Results...")
aggregated_ddf.to_csv(os.path.join(OUT_FOLDER,'./processed',f'cells_{cell_size}.csv'), index=False)

print("Done!")
