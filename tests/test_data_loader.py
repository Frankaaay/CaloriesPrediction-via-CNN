import os
import torch
import pandas as pd
import pytest
import sys
from PIL import Image

# Add src to path so we can import from data_loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_loader import FoodDataset, get_default_transforms

@pytest.fixture
def mock_dataset(tmp_path):
    """Create a mock dataset directory with dummy images and metadata."""
    # Create mock CSV
    metadata_csv = tmp_path / "mock_nutrition_data.csv"
    mock_data = pd.DataFrame([
        ["dish_1", 500, 0, 0, 0, 0, 0, 0],
        ["dish_2", 800, 0, 0, 0, 0, 0, 0]
    ])
    mock_data.to_csv(metadata_csv, header=False, index=False)

    # Create mock images
    dataset_dir = tmp_path / "realsense_overhead"
    dataset_dir.mkdir()
    
    img1 = Image.new('RGB', (100, 100), color = 'red')
    img1.save(dataset_dir / "dish_1.png")
    
    img2 = Image.new('RGB', (100, 100), color = 'blue')
    img2.save(dataset_dir / "dish_2.png")

    return dataset_dir, metadata_csv

def test_data_loader_structure(mock_dataset):
    dataset_dir, metadata_csv = mock_dataset
    
    transform = get_default_transforms()
    dataset = FoodDataset(root_dir=str(dataset_dir), metadata_csv=str(metadata_csv), transform=transform)
    
    assert len(dataset) == 2
    
    img, label = dataset[0]
    
    # Assert shape is transformed to [3, 225, 225]
    assert img.shape == torch.Size([3, 225, 225])
    # Assert label is valid
    assert not torch.isnan(label)
    assert float(label) == 500.0

    img2, label2 = dataset[1]
    assert float(label2) == 800.0
