import pandas as pd
import numpy as np
import os
import asyncio
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm

# Main Execution
DATA_FOLDER = '/home/ign.fr/llandrieu/Documents/data/plonk/mapillary_full'
cell_size  = 10
compute_size = False

TOKEN = "MLY|6237607803001040|1e22d1eebebaa36be00be1a63a809e5a"
PROXY = "http://proxy.ign.fr:3128"

semaphores = asyncio.Semaphore(500)

#west, south, east, north = -180,-90,180,90,#latitude-1,longitude-1,latitude+1,longitude+1
#get_images_ids_url = f"https://graph.mapillary.com/images?access_token={TOKEN}&fields=id,geometry,altitude,compass_angle,computed_geometry,computed_altitude,computed_compass_angle,camera_type,thumb_original_url,thumb_256_url,height,width,sequence,captured_at&bbox={west},{south},{east},{north}&limit=2000"
#get_images_ids_url = f"https://graph.mapillary.com/{image_id}?access_token={TOKEN}&fields=id,geometry,computed_geometry,camera_type,height,width,sequence,captured_at"


async def process_one_id(onerow, session, semaphore, progress):
    async with semaphore:
        #print(onerow.name)
        image_id = int(onerow['image_id'])
        get_images_ids_url = f"https://graph.mapillary.com/{image_id}?access_token={TOKEN}&fields=id,geometry,computed_geometry,camera_type,height,width,sequence,captured_at"
        async with session.get(get_images_ids_url, proxy=PROXY) as response:
            if response.status == 200:
                data = await asyncio.wait_for(response.json(), timeout=1200)
                await asyncio.sleep(0.0001)
                progress.update(1)
                try:
                    return image_id, data['camera_type']=='perspective'
                except KeyError:
                    pass
    progress.update(1)
    return image_id, False

async def main(df):
    semaphores = asyncio.Semaphore(500)
    async with aiohttp.ClientSession() as session:
        # Initialize the progress bar
        progress = async_tqdm(total=len(df), desc="Processing", dynamic_ncols=True, position=0, leave=True)

        tasks = [process_one_id(onerow, session, semaphores, progress) for _, onerow in df.iterrows()]
        results = await asyncio.gather(*tasks)

        progress.close()  # Close the progress bar
        
        results_dict = {image_id: is_perspective for image_id, is_perspective in results}

        df_filtered = df[df['image_id'].map(results_dict)].copy()

         # Count and print the number of perspective cameras
        num_perspective_cameras = sum(results_dict.values())
        print(f"Number of perspective cameras: {num_perspective_cameras}")

    return df_filtered



if __name__ == '__main__':
    print("Reading Cells...")

    df =  pd.read_csv(f'cells_{cell_size}.csv')

    print("Filtering...")
    df_filtered = asyncio.run(main(df))

    print("SAving...") 
    df_filtered.to_csv(f'cells_{cell_size}_perspectives.csv', index=False)