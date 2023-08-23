import reverse_geocoder
import asyncio
import pandas as pd


def add_city_country(df):

location = reverse_geocoder.search(
        [(lat, lon) for lat, lon in zip(df["latitude"], df["longitude"])]
    )



if __name__ == '__main__':

    print("Reading Cells...")

    df =  pd.read_csv(f'cells_{cell_size}_filtered.csv')
    df['image_id'] = df['image_id'].astype(int)

    print("Adding Country & Cities..")
    add_city_country(df)

    print("Done!")