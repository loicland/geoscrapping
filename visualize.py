import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
from scipy.ndimage.filters import gaussian_filter
import os

cell_size  = 10
DATA_FOLDER = '/var/data/llandrieu/mapillary_full/'

def view_distribution(df, filename=''):
    latitude = df['latitude']
    longitude = df['longitude']

    # Create a figure with a map
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()

    #Define the bins for the histogram
    num_bins = 250
    lon_bins = np.linspace(longitude.min(), longitude.max(), num_bins)
    lat_bins = np.linspace(latitude.min(), latitude.max(), num_bins)

    # Calculate the 2D histogram
    hist, xedges, yedges = np.histogram2d(longitude, latitude, bins=[lon_bins, lat_bins])

    # Smooth the histogram
    #sigma = 1  # Width of the Gaussian filter
    #hist = gaussian_filter(hist, sigma)

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

    plt.show()
    plt.savefig(f'density_{cell_size}_{filename}.png')



if __name__ == '__main__':

    if True:

        print("Reading Cells...")

        df =  pd.read_csv(f'cells_{cell_size}_filtered.csv')
        df['image_id'] = df['image_id'].astype(int)

        print("Computing Density Map...")
        view_distribution(df)

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