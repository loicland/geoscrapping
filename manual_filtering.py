import os
import argparse
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='test or train')

    parser.add_argument('--fold', default='test')
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--rot', default=90)

    args = parser.parse_args()

    # Parameters
    args.csv_file_path = '/var/data/llandrieu/geoscrapping/processed/'+args.fold+'.csv'  # Replace with your CSV file path'
    args.image_folder = '/var/data/llandrieu/geoscrapping/images/'+args.fold+'/'      # Replace with your image folder path
    args.trash_folder = '/var/data/llandrieu/geoscrapping/images/quarantined/bad_'+args.fold+'_filtered'
    args.rot_folder = '/var/data/llandrieu/geoscrapping/images/quarantined/'

    manual_filtering(args)