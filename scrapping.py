import pandas as pd
import random
import numpy as np
import os

import dask.dataframe as dd
from dask.diagnostics import ProgressBar


#from utils import *

def lat_lon_to_grid_index(lat, lon, cell_size=10):
    lat_idx = (lat + 90) // (0.009 * cell_size) # 0.009° is approx 1km
    lon_idx = (lon + 180) // (0.009 * cell_size / np.cos(np.radians(lat)))
    return lat_idx.astype(int), lon_idx.astype(int)

# Main Execution
DATA_FOLDER = '/home/ign.fr/llandrieu/Documents/data/plonk/mapillary'
cell_size  = 1

# Set up Dask DataFrame
ordered_ddf = dd.read_csv(os.path.join(DATA_FOLDER, "*.csv"), assume_missing=True)

# Shuffle the Dask DataFrame, avoid 
ddf = ordered_ddf.sample(frac=1, random_state=42)

# This will give you a tuple of (Delayed, int)
print("Computing size of dataframe...")
shape = ddf.shape
num_rows = 0

#with ProgressBar():
#    num_rows = shape[0].compute()

print(f"Number of rows: {num_rows:,}")

# Convert lat/lon to grid indices
ddf["lat_idx"], ddf["lon_idx"] = lat_lon_to_grid_index(ddf["latitude"], ddf["longitude"], cell_size)

    # Compute resulting dataframe and get unique grid cells

print(f"Pruning on a {cell_size}km grid...")
with ProgressBar():
    # Group by lat_idx and lon_idx and aggregate to get one row for each group
    aggregated_ddf = ddf.groupby(["lat_idx", "lon_idx"]).agg({
        "image_id": "first",
        "latitude": "first",
        "longitude": "first"
    }).reset_index().compute()

shape = aggregated_ddf.shape
print(f"Selected {aggregated_ddf.shape[0]:,} cells")
num_rows = shape[0]
print(f"{num_rows} cells selected")

# Merge the unique_cells with the original DDF to get the full rows of the selected unique cells
 
print("Saving Results...")
with ProgressBar():
    aggregated_ddf.to_csv(f'cells_{cell_size}.csv', index=False) 

print("Done!")
