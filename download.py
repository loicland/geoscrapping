import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import os

cell_size = 0.1
MAX_RETRIES = 5
TIMEOUT_DURATION = 3600
CONCURRENT_REQUESTS = 32
SKIP_ROWS = int(0)
SPLIT = 'test'
OUT_FOLDER = '/var/data/llandrieu/geoscrapping/'
#OUT_FOLDER = '/home/ign.fr/llandrieu/Documents/code/geoscrapping/'
PROXY = "http://proxy.ign.fr:3128"


def is_valid_url(url):
    return isinstance(url, str) and url.startswith(('http://', 'https://'))

async def download_image(semaphore, session, url, id, image_path, tqdm_instance):
    async with semaphore:
        retries = 0
        while retries < MAX_RETRIES:
            try: 
                async with session.get(url, timeout=TIMEOUT_DURATION, proxy=PROXY) as response:
                    try:
                        if response.status != 200:
                            print(f"Error {response.status}: {id}, {url}")
                            break

                        content = content = await response.content.read()
                        img = Image.open(BytesIO(content))
                        img = img.convert("RGB")
                        width, height = img.size
                        if width >= 512 or height >= 512:
                            if width < height:
                                new_width = 512
                                new_height = int(height * (new_width / width))
                            else:
                                new_height = 512
                                new_width = int(width * (new_height / height))
                            img = img.resize(
                                (new_width, new_height), resample=Image.BILINEAR
                            )
                        img.save(image_path / f"{id}.jpg")
                        break
                    except (asyncio.TimeoutError, aiohttp.client_exceptions.ServerDisconnectedError, aiohttp.client_exceptions.ClientConnectorError) as e:
                        print(f"{type(e).__name__} - {id}, {url}")
                        retries += 1
                    except aiohttp.client_exceptions.ServerDisconnectedError:
                        print(f"Server disconnected error: {id}, {url}. Retry {retries+1}/5")
                        retries += 1
                        await asyncio.sleep(5)  # sleep for 5 seconds before retrying
                    except OSError:
                        print(f"OSError: {id}, {url}")
                        break
                    except aiohttp.client_exceptions.ClientPayloadError:
                        retries += 1
                        await asyncio.sleep(2)  # You can adjust the sleep duration
                    finally:
                        tqdm_instance.update(1)
            except aiohttp.client_exceptions.ServerDisconnectedError:
                print(f"Server disconnected error: {id}, {url}. Retry {retries+1}/5")
                retries += 1
                await asyncio.sleep(5)  # sleep for 5 seconds before retrying

    await asyncio.sleep(0.01)

async def download_images():
    # Read the csv file
    df = pd.read_csv(os.path.join(OUT_FOLDER,'processed',f'{SPLIT}_{cell_size}.csv'))
    image_path = Path(os.path.join(OUT_FOLDER,'images',f'{SPLIT}_{cell_size}'))
    image_path.mkdir(parents=True, exist_ok=True)

    df = df.iloc[SKIP_ROWS:]

    df["thumb_original_url"] = df["thumb_original_url"].astype(str)
    df = df[["image_id", "thumb_original_url"]]
    df = df.dropna(subset=["thumb_original_url"])

    tqdm_instance = tqdm(total=len(df))
    already_downloaded = set([int(x.stem) for x in image_path.glob("*.jpg")])
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        tasks = []
        for index, row in df.iterrows():
            url = row["thumb_original_url"]
            id = row["image_id"]
            if id in already_downloaded:
                tqdm_instance.update(1)
                continue
            if not is_valid_url(url):
                print(f"Skipping invalid URL for id {row['id']}: {url}")
                continue
            task = asyncio.ensure_future(
                download_image(
                    semaphore, session, url, id, image_path, tqdm_instance=tqdm_instance
                )
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_images())