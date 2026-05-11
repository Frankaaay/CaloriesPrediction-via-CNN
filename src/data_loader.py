import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FoodDataset(Dataset):
    """Dataset class for Nutrition5k calories prediction."""
    def __init__(self, root_dir, metadata_csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Load only first 8 columns as the original code did
        self.data = pd.read_csv(metadata_csv, header=None, usecols=range(8))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dish_id = self.data.iloc[idx].iloc[0]
        calories = self.data.iloc[idx].iloc[1]

        # In original data, dish_id might be float/int, parse it explicitly
        img_path = os.path.join(self.root_dir, f"{dish_id}.png")
        
        if not os.path.isfile(img_path):
            image = Image.new("RGB", (225, 225), (0, 0, 0))  # Black image
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(np.nan)  # NaN as an invalid label
            
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(calories, dtype=torch.float32)

def get_default_transforms():
    """Returns the default torchvision transforms for training/inference."""
    return transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.ToTensor(),
    ])
