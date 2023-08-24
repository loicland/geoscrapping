import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os
from visualize import view_distribution

cell_size = 0.1
TEST_SIZE = 250000
TRAIN_SIZE = 2500000
MIN_PER_CELL = 10
COLLISION_DIST = 1
OUT_FOLDER = '/var/data/llandrieu/geoscrapping/'

def split(df):
     
     # Sample data (replace this with your data)
    longitude = df["longitude"].values
    latitude = df["latitude"].values

    view_distribution(df, 'full', cell_size=cell_size)

    # 1. Create a 2D histogram
    num_bins = 100
    lon_bins = np.linspace(longitude.min(), longitude.max(), num_bins)
    lat_bins = np.linspace(latitude.min(), latitude.max(), num_bins)
    hist, xedges, yedges = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])

    # 2. Identify which points fall into each of the bins
    df['lon_bin'] = np.clip(np.digitize(df['longitude'], lon_bins) - 1, 0, num_bins - 2) # Note the -2
    df['lat_bin'] = np.clip(np.digitize(df['latitude'], lat_bins) - 1, 0, num_bins - 2) # Note the -2

    # Drop areas with too low density
    low_density_lon_bins, low_density_lat_bins = np.where(hist < MIN_PER_CELL)

    # Create a set of bins with low density
    low_density_bin_set = set(zip(low_density_lon_bins, low_density_lat_bins))

    # Filter dataframe by excluding rows with (lon_bin, lat_bin) in low_density_bin_set
    df = df[~df[['lon_bin', 'lat_bin']].apply(tuple, axis=1).isin(low_density_bin_set)]

    # 4. Sample points based on the inverse of the square root of the density
    weights = 1. / np.power(hist[df['lon_bin'], df['lat_bin']], 0.75)
    normalized_weights = weights / np.sum(weights)

    test_df = df.sample(n=TEST_SIZE, weights=normalized_weights, replace=False)

    view_distribution(test_df, filename='test', cell_size=cell_size) 

    # Drop the rows sampled in test_df from the main df before sampling for train_df
    df = df.drop(test_df.index)

    # Recompute the weights for the remaining data in df
    weights_remaining = 1. / np.power(hist[df['lon_bin'], df['lat_bin']], 0.75)
    normalized_weights_remaining = weights_remaining / weights_remaining.sum()

    # Now sample for the train set
    train_df = df.sample(n=TRAIN_SIZE, weights=normalized_weights_remaining, replace=False) 

    # Convert lat/long to radians for BallTree
    train_df_rad = np.radians(train_df[['latitude', 'longitude']])
    test_df_rad = np.radians(test_df[['latitude', 'longitude']])

    # Create a BallTree
    print("Building Tree...")
    tree = BallTree(train_df_rad, leaf_size=15, metric='haversine')

    # Query the tree for any point within 1km of test set points
    print("Removing Collisions...")
    ind, dist = tree.query_radius(test_df_rad, r=COLLISION_DIST/6371, return_distance=True)  # 1km/6371km (Earth's radius) to convert to radians

    # Flatten the index list and get unique values (set for O(1) complexity during lookup)
    to_remove = set(np.concatenate(ind))

    # Remove the collisions from train_df
    #train_df = train_df.drop(train_df.iloc[list(to_remove)].index)

    view_distribution(train_df, filename='train', cell_size=cell_size) 

    print(f'Removed {len(to_remove)} collisions')

    return test_df, train_df
     
if __name__ == "__main__":


    print("Reading Cells...")

    df =  pd.read_csv(os.path.join(OUT_FOLDER,'./processed',f'cells_{cell_size}_enriched.csv'))

    train_df, test_df = split(df) 

    print("Saving Data ..")

    train_df.to_csv(os.path.join(OUT_FOLDER,'./processed',f'train_{cell_size}.csv'), index=False)
    test_df.to_csv(os.path.join(OUT_FOLDER,'./processed',f'test_{cell_size}.csv'), index=False)

