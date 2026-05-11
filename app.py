import streamlit as st
import torch
from PIL import Image
import os
import sys

# Import our modularized logic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from data_loader import get_default_transforms
from model import get_model

st.set_page_config(page_title="AI Calorie Estimator", page_icon="🍔", layout="centered")

st.title("🍔 AI Calorie Estimator")
st.write("Upload an overhead image of a meal, and our ResNet50-based Convolutional Neural Network will estimate the total calories.")

@st.cache_resource
def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, None

# Try loading the model. If you haven't trained it yet, this handles it gracefully.
model_path = "model.pth"
model, device = load_model(model_path)

uploaded_file = st.file_uploader("Choose a food image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Analysis")
        if model is None:
            st.error(f"⚠️ Model weights (`{model_path}`) not found! Please train the model on Colab, download `model.pth`, and place it in the root directory.")
        else:
            with st.spinner("Analyzing image features..."):
                # Preprocess the image
                transform = get_default_transforms()
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    prediction = model(img_tensor).item()
                
                st.success("Analysis Complete!")
                st.metric(label="Estimated Calories", value=f"{prediction:.0f} kcal")
                
            st.info("💡 **How it works:** This model looks at the pixel density, colors, and textures of the food items to map visual features to a nutritional regression scale.")
