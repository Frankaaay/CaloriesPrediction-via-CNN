from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from model import get_model
from data_loader import get_default_transforms

app = FastAPI(title="Calories Prediction API", version="1.0")

# Load model globally on startup (Cold start)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()

# Path to the trained model weights
model_path = os.path.join(os.path.dirname(__file__), "../model.pth")

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model could not be loaded. Please ensure model.pth exists. Error: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Calories Prediction API! Send a POST request to /predict with an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, runs it through the ResNet50 model,
    and returns the predicted calories as JSON.
    """
    try:
        # Read image bytes and convert to PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Transform the image
        transform = get_default_transforms()
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            prediction = model(img_tensor).item()
            
        # Return the calorie count format as JSON
        return JSONResponse(content={
            "filename": file.filename, 
            "calories": round(prediction, 2)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    # When run directly, this exposes the API on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
