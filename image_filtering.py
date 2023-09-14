import csv
import random
import shutil
import os
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

#import vissl
#import torch

#from omegaconf import OmegaConf
#from vissl.utils.hydra_config import AttrDict
#from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict

#from vissl.models import build_model

#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
#from PIL import Image
#import pandas as pd

def make_consistent(args):
    "ensure the consistency between images and csv"
    df = pd.read_csv(args.csv_file_path)
    image_ids_from_csv = set(df['image_id'].astype(str))
    all_image_files = [f for f in os.listdir(args.image_folder) if f.endswith('.jpg')]

    unmatched_images = [f for f in all_image_files if f[:-4] not in image_ids_from_csv]

    # Move unmatched images to args.trash_folder
    for image in unmatched_images:
        source = os.path.join(args.image_folder, image)
        destination = os.path.join(args.trash_folder, image)
        shutil.move(source, destination)
    
    image_ids_from_folder = {f[:-4] for f in os.listdir(args.image_folder) if f.endswith('.jpg')}
    filtered_df = df[df['image_id'].astype(str).isin(image_ids_from_folder)]

    filtered_df.to_csv(args.csv_file_path, index=False)

def is_blurry(image_path, threshold=120):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    try: 
        fft = np.fft.fft2(image)
    except IndexError:
        return True
    fftshift = np.fft.fftshift(fft)
    magnitude_spectrum = 20*np.log(np.abs(fftshift))
    return np.mean(magnitude_spectrum) < threshold

def is_dark(image_path, threshold=50):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean_brightness = np.mean(image)
    return mean_brightness < threshold

def create_canary(args):

    # Ensure canary directory exists
    if not os.path.exists(canary_folder):
        os.makedirs(canary_folder)

    # Read ids from the CSV
    all_ids = []
    with open(args.csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            all_ids.append(row["image_id"])

    # Randomly select 1000 ids
    random.seed(1)
    selected_ids = random.sample(all_ids, 1010)

    # Move the corresponding images to the canary folder
    for img_id in selected_ids:
        image_path = os.path.join(args.image_folder, f"{img_id}.jpg")
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(canary_folder, f"{img_id}.jpg"))
        else:
            print(img_id)

    print("Process completed!")

def filtered_out(img_path):
    #return is_dark(img_path)
    #return is_blurry(img_path)
    return is_blurry(img_path) or is_dark(img_path)

def filter_folder(args):
    # Get list of images
    images = os.listdir(args.image_folder)
    images = images[args.skip:]

    n_killed = 0
    n_seen = 0

    for img_file in tqdm(images, desc="Filtering images", unit="file"):
        n_seen = n_seen+1
        img_path = os.path.join(args.image_folder, img_file)

        # Check if the image should be dismissed
        if filtered_out(img_path):
            # Move it to the args.trash_folder
            trash_path = os.path.join(args.trash_folder, img_file)
            shutil.move(img_path, trash_path)
            n_killed = n_killed + 1
            print(f"Killed image {n_seen}, {n_killed} images killed")


def evaluate_canary(args):
    good_canary_images = [os.path.join(args.good_canary_folder, img) for img in os.listdir(args.good_canary_folder) if img.endswith('.jpg')]
    bad_canary_images = [os.path.join(args.bad_canary_folder, img) for img in os.listdir(args.bad_canary_folder) if img.endswith('.jpg')]

    TP, FP, FN, TN = 0, 0, 0, 0
    print("FN : ")
    str = ""
    # Check good_canary
    for img_path in good_canary_images:
        if filtered_out(img_path):
            print(img_path)
            str = str + " " + img_path
            FN += 1  # Should be good, but identified as bad
        else:
            TP += 1  # Correctly identified as good
    print("TN : ")
    print(str)
    str = ""
    # Check bad_canary
    for img_path in bad_canary_images:
        if filtered_out(img_path):
            print(img_path)
            str = str + " " + img_path
            TN += 1  # Correctly identified as bad
        else:
            FP += 1  # Should be bad, but identified as good
    print(str)

    print("Confusion Matrix:")
    print(f"TP: {TP}, FP: {FP}")
    print(f"FN: {FN}, TN: {TN}")

    IoU = TP / (TP + FP + FN)
    print(f"IoU: {IoU:.2f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='test or train')

    parser.add_argument('--fold', default='train')
    parser.add_argument('--skip', type=int, default=0)

    args = parser.parse_args()

        # Parameters
    args.csv_file_path = '/var/data/llandrieu/geoscrapping/processed/'+args.fold+'.csv'  # Replace with your CSV file path'
    args.image_folder = '/var/data/llandrieu/geoscrapping/images/'+args.fold+'/'      # Replace with your image folder path
    args.good_canary_folder = '/var/data/llandrieu/geoscrapping/images/canary/'
    args.bad_canary_folder = '/var/data/llandrieu/geoscrapping/images/bad_canary/'
    args.trash_folder = '/var/data/llandrieu/geoscrapping/images/trash/bad_'+args.fold+'_filtered'

    filter_folder(args)
    #make_consistent(args)
    #create_canary(args)
    #evaluate_canary(args)

    

    

 
