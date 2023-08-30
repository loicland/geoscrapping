import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os
from visualize import view_distribution

cell_size = 0.1
TEST_SIZE = 500000
TRAIN_SIZE = 5000000
MIN_UNIQUE_SEQUENCES_PER_CELL = 5
MIN_PER_CELL = 10
COLLISION_DIST = 1
OUT_FOLDER = '/var/data/llandrieu/geoscrapping/'

def split(df):

    print("Filtering First")
     
    # Sample data (replace this with your data)
    longitude = df["longitude"].values
    latitude = df["latitude"].values

    #view_distribution(df, 'full', cell_size=cell_size)

    # 1. Create a 2D histogram
    num_bins = 100
    lon_bins = np.linspace(longitude.min(), longitude.max(), num_bins)
    lat_bins = np.linspace(latitude.min(), latitude.max(), num_bins)

    # 2. Identify which points fall into each of the bins
    df['lon_bin'] = np.clip(np.digitize(df['longitude'], lon_bins) - 1, 0, num_bins - 2) # Note the -2
    df['lat_bin'] = np.clip(np.digitize(df['latitude'], lat_bins) - 1, 0, num_bins - 2) # Note the -2

    # 2. Identify cells with too few unique sequences
    initial_rows = len(df)
    broadcasted_sequence_counts = df.groupby(['lon_bin', 'lat_bin'])['sequence'].transform('nunique')
    df = df[broadcasted_sequence_counts >= MIN_UNIQUE_SEQUENCES_PER_CELL]

    print(f"Removed {initial_rows - len(df):,}  imgs for having too few sequences")

    #recompute distribution 
    longitude = df["longitude"].values
    latitude = df["latitude"].values
    hist, _, _ = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])

    # Drop areas with too low density
    low_density_lon_bins, low_density_lat_bins = np.where(hist < MIN_PER_CELL)
    low_density_bin_set = set(zip(low_density_lon_bins, low_density_lat_bins))
    mask = df.set_index(['lon_bin', 'lat_bin']).index.isin(low_density_bin_set)
    df = df[~mask]

    print(f"Removed {len(low_density_bin_set):,} cells for having too few images")

    view_distribution(df, 'full_filtered', cell_size=cell_size)

    # 4. Sample points based on the inverse of the square root of the density

    #recompute distribution 
    longitude = df["longitude"].values
    latitude = df["latitude"].values
    df['lon_bin'] = np.clip(np.digitize(df['longitude'], lon_bins) - 1, 0, num_bins - 2) # Note the -2
    df['lat_bin'] = np.clip(np.digitize(df['latitude'], lat_bins) - 1, 0, num_bins - 2) # Note the -2

    hist, _, _ = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])
    weights = 1. / np.power(hist[df['lon_bin'], df['lat_bin']], 0.75)
    normalized_weights = weights / np.sum(weights)

    test_df = df.sample(n=TEST_SIZE, weights=normalized_weights, replace=False, random_state=42)

    view_distribution(test_df, filename='test', cell_size=cell_size) 

    # Drop the rows sampled in test_df from the main df before sampling for train_df
    df = df.drop(test_df.index)

    # Remove all rows from df that have a sequence present in the test set
    initial_rows = len(df)
    test_sequences = set(test_df['sequence'].values)
    df = df[~df['sequence'].isin(test_sequences)]
    print(f"Removed {initial_rows - len(df):,} images due to sequence overlap with the test set.")

    # We now remove all points that are too close to the test set

    # Create a BallTree
    print("Building Tree...")
    df_rad = np.radians(df[['latitude', 'longitude']])
    test_df_rad = np.radians(test_df[['latitude', 'longitude']])
    tree = BallTree(df_rad, leaf_size=15, metric='haversine')

    # Query the tree for any point within 1km of test set points
    print("Removing Collisions...")
    ind, _ = tree.query_radius(test_df_rad, r=COLLISION_DIST/6371, return_distance=True)  # 1km/6371km (Earth's radius) to convert to radians

    # Flatten the index list and get unique values (set for O(1) complexity during lookup)
    to_remove = set(np.concatenate(ind))
    mask = ~df.index.isin(to_remove)
    df = df[mask]

    print(f"Removed {len(to_remove)} images for being too close to the test set")

    view_distribution(df, 'full_filtered2', cell_size=cell_size)

    # Recompute the weights for the remaining data in df
    longitude = df["longitude"].values
    latitude = df["latitude"].values
    df['lon_bin'] = np.clip(np.digitize(df['longitude'], lon_bins) - 1, 0, num_bins - 2) # Note the -2
    df['lat_bin'] = np.clip(np.digitize(df['latitude'], lat_bins) - 1, 0, num_bins - 2) # Note the -
    weights_remaining = 1. / np.power(hist[df['lon_bin'], df['lat_bin']], 0.75)
    normalized_weights_remaining = weights_remaining / weights_remaining.sum()
    
    # Now sample for the train set
    train_df = df.sample(n=TRAIN_SIZE, weights=normalized_weights_remaining, replace=False, random_state=42) 

    view_distribution(train_df, filename='train', cell_size=cell_size) 

    print(f'Removed {len(to_remove)} collisions')

    return train_df, test_df
     
if __name__ == "__main__":


    print("Reading Cells...")

    df =  pd.read_csv(os.path.join(OUT_FOLDER,'./processed',f'cells_{cell_size}_enriched.csv'))

    df = df.dropna(subset=['image_id', 'latitude', 'longitude', 'thumb_original_url', 'country', 'sequence', 'captured_at'])

    train_df, test_df = split(df) 

    print("Saving Data ..")

    train_df.to_csv(os.path.join(OUT_FOLDER,'./processed',f'train2_{cell_size}.csv'), index=False)
    test_df.to_csv(os.path.join(OUT_FOLDER,'./processed',f'test_{cell_size}.csv'), index=False)

