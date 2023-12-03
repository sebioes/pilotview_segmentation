import torch
from torchvision import transforms
import numpy as np
import glob
import cv2
import os
import argparse
from tqdm import tqdm
import kornia.augmentation as K

class Augment():
    def __init__(self):
        self.brightness = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        self.contrast = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        self.snow = K.RandomSnow(p=1.0, snow_coefficient=(0.5, 0.5), brightness=(2, 2))
        self.motion_blur = K.RandomMotionBlur(p=1.0, kernel_size=(6, 6), angle=(-60, 60), direction=(-1, 1))

    def night(self, image):
        brightness_factor = (0, 0.5) 
        self.brightness.brightness = brightness_factor

        contrast_factor = (1.5, 2.5)
        self.contrast.contrast = contrast_factor

        image = self.brightness(image)
        image = self.contrast(image)

        return image
    
    def snow(self, image):
        image = self.snow(image)
        return image
    
    def bright(self, image):
        brightness_factor = (1.5, 2.0)
        self.brightness.brightness = brightness_factor
        image = self.brightness(image)
        return image

    def blur(self, image):
        image = self.motion_blur(image)
        return image

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", type=str, default=None, help="augmentation type")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="dataset directory")
    args = parser.parse_args()

    # make directory
    aug_dict = {"night": Augment().night, "snow": Augment().snow, "bright": Augment().bright, "blur": Augment().blur}
    assert args.augment is not None, f"Please specify augmentation type from {list(aug_dict.keys())}"

    # import all images from train directory
    # glob for jpg, jpeg, png
    image_paths = glob.glob(os.path.join(args.dataset_dir, "train/*.jpg")) \
                + glob.glob(os.path.join(args.dataset_dir, "train/*.jpeg")) \
                + glob.glob(os.path.join(args.dataset_dir, "train/*.png"))
    out_dir = f"{args.dataset_dir}/train_augmented_{args.augment}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for path in tqdm(image_paths, desc="Augmenting images", total=len(image_paths)):
        # load image
        image = cv2.imread(path)
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert to tensor
        image = transforms.ToTensor()(image)
        # image = augment(image)
        image = aug_dict[args.augment](image)

        # save
        if len(image.shape) == 4: # if there is batch dimension
            image = image.squeeze(0) # remove batch dimension
        image = transforms.ToPILImage()(image)
        image.save(out_dir + path.split("/")[-1].split(".")[0] + f"_augmented_{args.augment}.jpg")
        