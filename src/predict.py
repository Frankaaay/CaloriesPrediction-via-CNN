import torch
from PIL import Image
import argparse

from data_loader import get_default_transforms
from model import get_model

def predict_single_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init model
    model = get_model()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return
        
    model.to(device)
    model.eval()
    
    # Process image
    transform = get_default_transforms()
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return
        
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dim
    
    # Predict
    with torch.no_grad():
        prediction = model(img_tensor).item()
        
    print(f"Image: {image_path}")
    print(f"Predicted Calories: {prediction:.2f} kcal")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Calories from an Image")
    parser.add_argument("image_path", type=str, help="Path to the food image")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to the trained model weights (.pth)")
    
    args = parser.parse_args()
    predict_single_image(args.image_path, args.model_path)
