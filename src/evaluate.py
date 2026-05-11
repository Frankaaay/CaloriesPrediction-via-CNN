import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import argparse
import os
import sys

# Ensure src modules can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data_loader import FoodDataset, get_default_transforms
from model import get_model

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load Model
    model = get_model()
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded {args.model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    model.to(device)
    model.eval()

    # Load Data
    transform = get_default_transforms()
    try:
        dataset = FoodDataset(args.data_dir, args.metadata, transform=transform)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Use a fixed seed to get a consistent test split every time
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size = int(0.8 * len(dataset))
    test_indices = indices[train_size:]
    test_dataset = Subset(dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Running evaluation on test set... This might take a moment.")
    mae, mse, count = 0.0, 0.0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            mask = ~torch.isnan(labels)
            if mask.sum() == 0:
                continue

            images, labels = images[mask].to(device), labels[mask].to(device)
            
            outputs = model(images).squeeze()
            # Handle single item batches where squeeze removes the batch dimension
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            mae += torch.abs(outputs - labels).sum().item()
            mse += torch.sum((outputs - labels) ** 2).item()
            count += labels.size(0)

    if count > 0:
        mae /= count
        rmse = np.sqrt(mse / count)
        print(f"--- Evaluation Results ---")
        print(f"Total Test Images: {count}")
        print(f"Mean Absolute Error (MAE): {mae:.2f} kcal")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kcal")
    else:
        print("No valid labels found for evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Calories CNN Model")
    parser.add_argument("--data_dir", type=str, default="realsense_overhead/", help="Path to image dataset")
    parser.add_argument("--metadata", type=str, default="metadata/nutrition_data.csv", help="Path to metadata CSV")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to saved model weights")
    args = parser.parse_args()
    
    evaluate(args)
