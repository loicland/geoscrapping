import reverse_geocoder
import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, asin
import matplotlib.pyplot as plt

cell_size = 0.1
OUT_FOLDER = '/var/data/llandrieu/geoscrapping/'


def haversine_distance(lat1, lon1, lat2, lon2):
    if (lat1 is None) or (lon1 is None) or (lat2 is None) or (lon2 is None):
        return 0
    R = 6371  # radius of the earth in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = (
        sin(dLat / 2.0) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2.0) ** 2
    )
    c = 2 * asin(sqrt(a))
    distance = R * c
    return distance


if __name__ == '__main__':

    print("Reading Cells...")

    #df =  pd.read_csv(os.path.join(OUT_FOLDER,'./processed',f'cells_{cell_size}_filtered.csv'))
    df =  pd.read_csv(os.path.join(OUT_FOLDER,'processed',f'train_{cell_size}.csv'))
    df['image_id'] = df['image_id'].astype(int)

    print("Adding Country & Cities..")
    location = reverse_geocoder.search(
            [(lat, lon) for lat, lon in zip(df["latitude"], df["longitude"])]
        )
    
    lat_city =  [float(l.get("lat", "")) for l in location]
    lon_city =  [float(l.get("lon", "")) for l in location]

    df["country"] = [l.get("cc", "") for l in location]
    df["admin1"] = [l.get("admin1", "") for l in location]
    df["admin2"] = [l.get("admin2", "") for l in location]
    df["city"] = [l.get("name", "") for l in location]

    print("Filtering by distance to city")
    #compute distance to nearest city
    d = np.array([
        haversine_distance(lat1, lon1, lat2, lon2) 
        for lat1, lon1, lat2, lon2 in zip(df["latitude"], df["longitude"], lat_city, lon_city)
        ])

    # Create a mask where distances are less than or equal to 100km
    mask = d <= 100

    # Apply the mask to the dataframe
    df_filtered = df[mask].copy()

    #print("Filtering by image count per country")

    #country_counts = df_filtered['country'].value_counts()
    #filtered_countries = country_counts[country_counts >= 2].index
    #df_filtered = df[df['country'].isin(filtered_countries)]

    print("Saving...") 
    df_filtered.to_csv(os.path.join(OUT_FOLDER,'processed',f'train_{cell_size}.csv'), index=False)