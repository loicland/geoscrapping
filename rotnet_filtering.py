import vissl
import torch
import shutil
import os
from tqdm import tqdm
from pathlib import Path
import argparse

from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict

from vissl.models import build_model

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

def filtering(args):

    device = torch.device("cpu")#"cuda:0" if torch.cuda.is_available() else "cpu")

    # Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
    # All other options override the simclr_8node_resnet.yaml config.

    cfg = [
        "config=pretrain/rotnet/rotnet_8gpu_resnet.yaml",
        "config.CHECKPOINT.DIR=weights",  # Specify path for the model weights.
    ]

    # Compose the hydra configuration.
    cfg = compose_hydra_configuration(cfg)
    # Convert to AttrDict. This method will also infer certain config options
    # and validate the config is valid.
    _, cfg = convert_to_attrdict(cfg)

    model = build_model(cfg.MODEL, cfg.OPTIMIZER)

    # Load the weights.
    checkpoint = torch.load(
        "weights/converted_vissl_rn50_rotnet_in22k_ep105.torch", map_location="cpu"
    )
    checkpoint_model = {
        f"{key_base}.{key}": value
        for key_base in checkpoint["classy_state_dict"]["base_model"]["model"].keys()
        for key, value in checkpoint["classy_state_dict"]["base_model"]["model"][
            key_base
        ].items()
    }
    # Load the weights.
    model.load_state_dict(checkpoint_model)

    # Move the model to the device.
    model.to(device)
    # Set the model to evaluation mode.
    model.eval()


    class DatasetWithPath(Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.imgs = list(sorted(os.listdir(root)))[args.skip:]

        def __getitem__(self, idx):
            path = os.path.join(self.root, self.imgs[idx])
            img = Image.open(path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, path

        def __len__(self):
            return len(self.imgs)


    dataset = DatasetWithPath(
        args.image_folder,
        transform=transforms.Compose(
            [
                transforms.CenterCrop(512),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rotations = [0,90,180,270]

    for i, (x, path) in enumerate(tqdm(dataloader)):

        modified_paths = [
            path.replace('/images/'+args.fold, '/images/quarantined/'+args.fold+'_180')
            for path in path
        ]

        existing_modified_paths = [
            modified_path for modified_path in modified_paths
            if os.path.exists(modified_path)
        ]

        if len(existing_modified_paths)>0:
            print(existing_modified_paths)
            continue



        x = x.to(device)
        with torch.no_grad():
            y = torch.nn.functional.softmax(model(x)[0], dim=1).argmax(dim=1)
        for j in range(len(path)):
            if y[j] != 0:
                print(rotations[y[j]])
                image_id = path[j].split("/")[-1].split(".")[0]
                destination_path = os.path.join(args.rot_folder, args.fold + "_" + str(rotations[y[j]]), image_id + ".jpg")

                if os.path.exists(destination_path):
                    print(f"File {destination_path} already exists!")
                else:
                    shutil.copy(path[j], destination_path)

def rotate(args):
    rotations = [180,270]
    for rot in rotations:
        folder_path = args.rot_folder+args.fold+"_"+str(rot)+"/"
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                filepath = os.path.join(folder_path, filename)
            # Open the image
            with Image.open(filepath) as img:
                # Rotate and save the image back
                img_rotated = img.rotate(rot, expand=True)  # negative value to rotate clockwise
                img_rotated.save(filepath)
    return


def update_rotation(args):
    rotations = [90,180,270]
    for rot in rotations:
        folder_path = args.rot_folder+args.fold+"_"+str(rot)+"/"
        for img in os.listdir(folder_path):
            rotate_img_path = os.path.join(folder_path, img)
            root_img_path = os.path.join(args.image_folder , img)
            
            # Check if it's a file before moving
            if os.path.isfile(rotate_img_path):
                shutil.copy(rotate_img_path, root_img_path)
            else:
                print("!!!" + rotate_img_path)

def purge(args):
    bad_folder_path = args.rot_folder+args.fold+"_bad/"
    for img in os.listdir(bad_folder_path):
        bad_img_path = os.path.join(bad_folder_path, img)
        root_img_path = os.path.join(args.image_folder , img)
            
        # Check if it's a file before moving
        if os.path.isfile(root_img_path):
            os.remove(root_img_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='test or train')

    parser.add_argument('--fold', default='train')
    parser.add_argument('--skip', type=int, default=2000000)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

        # Parameters
    args.csv_file_path = '/var/data/llandrieu/geoscrapping/processed/'+args.fold+'.csv'  # Replace with your CSV file path'
    args.image_folder = '/var/data/llandrieu/geoscrapping/images/'+args.fold+'/'      # Replace with your image folder path
    args.trash_folder = '/var/data/llandrieu/geoscrapping/images/quarantined/bad_'+args.fold+'_filtered'
    args.rot_folder = '/var/data/llandrieu/geoscrapping/images/quarantined/'

    #rotate(args)
    filtering(args)
    #update_rotation(args)
    #purge(args)
    #create_canary(args)
    #evaluate_canary(args)