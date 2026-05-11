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
        
        # The CSV has a variable number of columns depending on the number of ingredients.
        # We declare 'names=range(200)' to force Pandas to parse up to 200 columns safely,
        # but 'usecols=[0, 1]' selectively keeps only dish_id and calories.
        self.data = pd.read_csv(metadata_csv, header=None, names=range(200), usecols=[0, 1], low_memory=False)
        self._missing_printed = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Force string & strip to avoid whitespace or weird formatting
        dish_id = str(self.data.iloc[idx, 0]).strip()
        
        try:
            calories = float(self.data.iloc[idx, 1])
        except (ValueError, TypeError):
            calories = np.nan

        # The dataset has folders named dish_ID, containing rgb.png
        img_path = os.path.join(self.root_dir, dish_id, "rgb.png")
        
        if not os.path.isfile(img_path):
            if self._missing_printed < 3:
                print(f"⚠️ DEBUG Warning: Could not find image at path: {img_path}")
                self._missing_printed += 1
                
            image = Image.new("RGB", (225, 225), (0, 0, 0))  # Black image
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
