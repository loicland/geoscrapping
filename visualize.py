import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
from scipy.ndimage.filters import gaussian_filter
import os

cell_size  = 0.1
DATA_FOLDER = '/var/data/llandrieu/mapillary_full/'
OUT_FOLDER = '/var/data/llandrieu/geoscrapping/'

def compute_entropy(series):
    # Compute probabilities
    p = series.value_counts(normalize=True)
    # Compute entropy
    entropy = -np.sum(p * np.log(p))
    # Number of unique classes or categories
    n_unique = len(p)
    # Normalize the entropy
    normalized_entropy = entropy / np.log(n_unique)
    return normalized_entropy

def view_distribution(df, filename='', cell_size=cell_size):
    
    country_counts = df['country'].value_counts().head(20)
    country_percentage = country_counts / len(df) * 100
    print(country_percentage)

    latitude = df['latitude']
    longitude = df['longitude']

    # Create a figure with a map
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()

    #Define the bins for the histogram
    num_bins = 100
    lon_bins = np.linspace(longitude.min(), longitude.max(), num_bins)
    lat_bins = np.linspace(latitude.min(), latitude.max(), num_bins)

    df['lon_bin'] = np.clip(np.digitize(df['longitude'], lon_bins) - 1, 0, num_bins - 2) # Note the -2
    df['lat_bin'] = np.clip(np.digitize(df['latitude'], lat_bins) - 1, 0, num_bins - 2) # Note the -2

    country_entropy = compute_entropy(df['country'])
    df['cell'] = list(zip(df['lon_bin'], df['lat_bin']))
    cell_gini = compute_entropy(df['cell'])
    print(f"Country entropy: {country_entropy:.3f} + cell entropy: {cell_gini:.3f}")

    # Calculate the 2D histogram
    hist, xedges, yedges = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])

    # Smooth the histogram
    sigma = 1  # Width of the Gaussian filter
    hist = gaussian_filter(hist, sigma)

    # Generate a logarithmic color scale
    cmap = plt.get_cmap('inferno')
    cmap.set_under('white', alpha=0)  # All densities below `low_density` are set to be fully transparent.
    low_density = 1
    norm = colors.LogNorm(vmin=low_density, vmax=hist.max())

    # Plot the density heatmap
    img = ax.pcolormesh(lon_bins, lat_bins, hist.T, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), rasterized=True)

    # Add a colorbar
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height/1])
    cb = fig.colorbar(img, cax=cax, extend='min')
    cb.set_label('Log-scaled count')

    #plt.show()    
    plt.savefig(os.path.join('./imgs',f'density_{cell_size}_{filename}.png'))


def view_distribution_simple(df, filename='', cell_size=cell_size):
    
    latitude = df['latitude']
    longitude = df['longitude']

    # Create a figure with a map
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()

    #Define the bins for the histogram
    num_bins = 400
    lon_bins = np.linspace(longitude.min(), longitude.max(), num_bins)
    lat_bins = np.linspace(latitude.min(), latitude.max(), num_bins)

    df['lon_bin'] = np.clip(np.digitize(df['longitude'], lon_bins) - 1, 0, num_bins - 2) # Note the -2
    df['lat_bin'] = np.clip(np.digitize(df['latitude'], lat_bins) - 1, 0, num_bins - 2) # Note the -2

      # Calculate the 2D histogram
    hist, xedges, yedges = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])

    # Smooth the histogram
    sigma = 0.5  # Width of the Gaussian filter
    hist = gaussian_filter(hist, sigma)

    # Generate a logarithmic color scale
    cmap = plt.get_cmap('inferno')
    cmap.set_under('white', alpha=0)  # All densities below `low_density` are set to be fully transparent.
    low_density = 1
    norm = colors.LogNorm(vmin=low_density, vmax=hist.max())

    # Plot the density heatmap
    img = ax.pcolormesh(lon_bins, lat_bins, hist.T, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), rasterized=True)

    # Add a colorbar
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height/1])
    cb = fig.colorbar(img, cax=cax, extend='min')
    cb.set_label('Log-scaled count')

    #plt.show()    
    plt.savefig(os.path.join('./imgs',f'density_{cell_size}_{filename}.png'))



if __name__ == '__main__':

    if True:

        print("Reading Cells...")

        #df =  pd.read_csv(os.path.join(OUT_FOLDER,'processed/',f'cells_{cell_size}_enriched.csv'))

        df =  pd.read_csv(os.path.join(OUT_FOLDER,f'./processed/train_{cell_size}.csv'))
        df['image_id'] = df['image_id'].astype(int)

        print("Computing Density Map...")
        view_distribution_simple(df)

        print("Done!")
    
    if False:
        csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(csv_file)
            
            file_path = os.path.join(DATA_FOLDER, csv_file)
    
            # Load the CSV into a DataFrame
            df = pd.read_csv(file_path)
    
            # Sample 100k rows randomly
            sample_df = df.sample(n=100000, replace=True) if len(df) > 100000 else df
    
            # Visualize the sampled data
            view_distribution(sample_df, csv_file)