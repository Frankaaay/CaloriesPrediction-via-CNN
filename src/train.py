import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import argparse

from data_loader import FoodDataset, get_default_transforms
from model import get_model

def train(args):
    # Configuration
    dataset_path = args.data_dir
    metadata_path = args.metadata
    
    config = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_epochs': args.epochs,
        'dropout_rate': 0.3,
        'optimizer': 'AdamW',
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Dataset & Dataloader
    transform = get_default_transforms()
    try:
        dataset = FoodDataset(dataset_path, metadata_path, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size = int(0.8 * len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, drop_last=True)

    # Model definition
    model = get_model(dropout_rate=config['dropout_rate'])
    model.to(device)

    # Optimizer & Loss
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

    # Training Loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        valid_batches = 0
        
        for images, labels in train_loader:
            mask = ~torch.isnan(labels)
            images, labels = images[mask], labels[mask]

            if len(labels) == 0:
                continue

            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            valid_batches += 1

        if valid_batches > 0:
            print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {running_loss/valid_batches:.4f}")

    # Evaluation
    model.eval()
    mae, mse, count = 0.0, 0.0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            mask = ~torch.isnan(labels)
            if mask.sum() == 0:
                continue

            images, labels = images[mask], labels[mask]

            outputs = model(images).squeeze()
            mae += torch.abs(outputs - labels).sum().item()
            mse += torch.sum((outputs - labels) ** 2).item()
            count += labels.size(0)

    if count > 0:
        mae /= count
        rmse = np.sqrt(mse / count)
        print(f"Test MAE: {mae:.2f}, Test RMSE: {rmse:.2f}, Test MSE: {mse:.2f}")

    # Save Model Weights
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Calories CNN")
    parser.add_argument("--data_dir", type=str, default="realsense_overhead/", help="Path to image dataset")
    parser.add_argument("--metadata", type=str, default="metadata/nutrition_data.csv", help="Path to metadata CSV")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="model.pth", help="Path to save the model weights")
    args = parser.parse_args()
    
    train(args)
