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
from tkinter import Tk, Label, PhotoImage
from PIL import Image, ImageTk




class ImageSorter:
    def __init__(self, args, master, images):
        self.master = master
        self.images = images
        self.source_folder = args.source_folder
        self.trash_folder = args.trash_folder
        self.index = 0

        # Load initial image
        self.load_image()

        # Bind keys
        self.master.bind("<space>", self.next_image)
        self.master.bind("x", self.move_image)

    def load_image(self):
        image_path = os.path.join(self.source_folder, self.images[self.index])
        pil_image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.label = Label(self.master, image=self.tk_image)
        self.label.pack()

    def next_image(self, event):
        # Clear current image and load next
        self.label.pack_forget()
        self.index += 1
        if self.index < len(self.images):
            self.load_image()
        else:
            # Close the application if no images left
            self.master.quit()

    def move_image(self, event):
        # Move current image to the special folder
        current_image = os.path.join(self.source_folder, self.images[self.index])
        destination_path = os.path.join(self.trash_folder, self.images[self.index])
        os.rename(current_image, destination_path)
        # Load next image
        self.next_image(event)

def manual_filtering(args):
    args.source_folder = args.rot_folder+args.fold+"_"+str(args.rot)+"/"
    args.trash_folder = args.rot_folder+args.fold+"_bad"

    images = [f for f in os.listdir(args.source_folder) if f.endswith(('.jpg'))]
    root = Tk()
    app = ImageSorter(root, args, images)
    root.mainloop()


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
        x = x.to(device)
        with torch.no_grad():
            y = torch.nn.functional.softmax(model(x)[0], dim=1).argmax(dim=1)
        for j in range(len(path)):
            if y[j] != 0:
                print(rotations[y[j]])
                image_id = path[j].split("/")[-1].split(".")[0]
                shutil.copy(path[j], args.rot_folder+args.fold+"_"+str(rotations[y[j]])+"/"+image_id+".jpg")



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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='test or train')

    parser.add_argument('--fold', default='test')
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rot', default=90)

    args = parser.parse_args()

        # Parameters
    args.csv_file_path = '/var/data/llandrieu/geoscrapping/processed/'+args.fold+'.csv'  # Replace with your CSV file path'
    args.image_folder = '/var/data/llandrieu/geoscrapping/images/'+args.fold+'/'      # Replace with your image folder path
    args.trash_folder = '/var/data/llandrieu/geoscrapping/images/quarantined/bad_'+args.fold+'_filtered'
    args.rot_folder = '/var/data/llandrieu/geoscrapping/images/quarantined/'

    #rotate(args)
    manual_filtering(args)
    #filtering(args)
    #make_consistent(args)
    #create_canary(args)
    #evaluate_canary(args)