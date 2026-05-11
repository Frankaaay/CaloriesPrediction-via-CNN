import torch.nn as nn
from torchvision import models

def get_model(dropout_rate=0.3):
    """
    Returns a modified ResNet50 model for calorie regression.
    """
    # Use the default standard weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify the fully connected layer for single output (regression)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 1)
    )
    
    return model
