import pandas as pd
import numpy as np
import os
import asyncio
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm
from math import radians, sin, cos, sqrt, asin

# Main Execution
#DATA_FOLDER = '/home/ign.fr/llandrieu/Documents/data/plonk/mapillary_full'
DATA_FOLDER = '/var/data/llandrieu/mapillary_full/'
cell_size  = 10
compute_size = False

TOKEN = "MLY|6237607803001040|1e22d1eebebaa36be00be1a63a809e5a"
PROXY = "http://proxy.ign.fr:3128"

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

def grid_index_to_lat_lon(lat_idx, lon_idx, cell_size=10):
    # Reverse the latitude calculations
    south = lat_idx * (0.009 * cell_size) - 90
    north = (lat_idx + 1) * (0.009 * cell_size) - 90

    # Reverse the longitude calculations
    delta_lon = 0.009 * cell_size / np.cos(np.radians((south + north) / 2))
    west = lon_idx * delta_lon - 180
    east = (lon_idx + 1) * delta_lon - 180
    
    return west, south, east, north


semaphores = asyncio.Semaphore(500)

#west, south, east, north = -180,-90,180,90,#latitude-1,longitude-1,latitude+1,longitude+1
#get_images_ids_url = f"https://graph.mapillary.com/images?access_token={TOKEN}&fields=id,geometry,altitude,compass_angle,computed_geometry,computed_altitude,computed_compass_angle,camera_type,thumb_original_url,thumb_256_url,height,width,sequence,captured_at&bbox={west},{south},{east},{north}&limit=2000"
#get_images_ids_url = f"https://graph.mapillary.com/{image_id}?access_token={TOKEN}&fields=id,geometry,computed_geometry,camera_type,height,width,sequence,captured_at"


async def process_one_id(onerow, session, semaphore, progress):
    async with semaphore:
        #print(onerow.name)
        image_id = int(onerow['image_id'])
        latitude = onerow['latitude']
        longitude = onerow['longitude']
        lat_idx = onerow['lat_idx']
        lon_idx = onerow['lon_idx']

        get_images_ids_url = f"https://graph.mapillary.com/{image_id}?access_token={TOKEN}&fields=id,altitude,camera_type,thumb_original_url,thumb_256_url,height,width,sequence,captured_at"
        async with session.get(get_images_ids_url, proxy=PROXY) as response:
            progress.update(1)

            if response.status != 200:
                return image_id, None
            
            data = await asyncio.wait_for(response.json(), timeout=1200)
            timestamp= data["captured_at"] / 1000
            await asyncio.sleep(0.0001)
            
            #check if the image is perspective
            try:
                if data['camera_type'] != 'perspective':
                    return image_id, None
            except KeyError:
                return image_id, None

        return image_id, data

        #try to find an image with the same sequence
        west, south, east, north = grid_index_to_lat_lon(lat_idx, lon_idx) #latitude-1,longitude-1,latitude+1,longitude+1
        get_images_ids_url_nei = f"https://graph.mapillary.com/images?access_token={TOKEN}&fields=id,geometry,sequence,captured_at&bbox={west},{south},{east},{north}&limit=2000"
        async with session.get(get_images_ids_url_nei, proxy=PROXY) as response_nei:
            if response_nei.status != 200:
                return image_id, None
            data_neis = await asyncio.wait_for(response_nei.json(), timeout=1200)
            data_neis = data_neis["data"]

            for data_nei in data_neis:
                if data_nei["sequence"] != data ["sequence"]:
                    continue
                if data_nei["id"] == data ["id"]:
                    continue
                longitude_nei = data_nei["geometry"]["coordinates"][0]
                latitude_nei = data_nei["geometry"]["coordinates"][1]
                timestamp_nei = data_nei["captured_at"]/1000

                dist = haversine_distance(latitude, longitude, latitude_nei, longitude_nei)
                speed = np.abs(dist / (timestamp - timestamp_nei + 1e-10)) * 3600

                if speed>150:
                    print(f"weird speed: {speed:,} km/h")
                    return image_id, None
                else:
                    #print(f"good speed: {speed:,} km/h")
                    return image_id, data
            #print(f"not filtered")
            return image_id, data

async def main(df):
    semaphores = asyncio.Semaphore(500)
    async with aiohttp.ClientSession() as session:
        # Initialize the progress bar
        progress = async_tqdm(total=len(df), desc="Processing", dynamic_ncols=True, position=0, leave=True)

        tasks = [process_one_id(onerow, session, semaphores, progress) for _, onerow in df.iterrows()]
        results = await asyncio.gather(*tasks)

        progress.close()  # Close the progress bar
        
        results_dict = {image_id: data for image_id, data in results if data}
        results_df = pd.DataFrame(list(results_dict.values()))
        results_df['image_id'] = results_df['id'].astype(int)
        merged_df = pd.merge(df, results_df, on='image_id', how='inner')

         # Count and print the number of acceptable cameras
        print(f"Number of ok images: {len(merged_df):,}")

    return merged_df

if __name__ == '__main__':

    print("Reading Cells...")

    df =  pd.read_csv(f'cells_{cell_size}.csv')
    df['image_id'] = df['first_image_id'].astype(int)

    print("Filtering...")
    df_filtered = asyncio.run(main(df))

    print("Saving...") 
    df_filtered.to_csv(f'cells_{cell_size}_filtered.csv', index=False)