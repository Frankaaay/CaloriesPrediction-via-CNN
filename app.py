import os
import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="AI Calorie Estimator", page_icon="🍔", layout="centered")

st.title("🍔 AI Calorie Estimator")
st.write("Upload an overhead image of a meal, and our ResNet50-based backend API will estimate the total calories.")

# The FastAPI backend URL
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

uploaded_file = st.file_uploader("Choose a food image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Analysis")
        
        with st.spinner("Sending image to backend API..."):
            try:
                # We send the uploaded file as a standard HTTP POST request in a multipart/form-data payload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("API Response Received!")
                    st.metric(label="Estimated Calories", value=f"{data['calories']} kcal")
                else:
                    st.error(f"Backend API returned an error: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to the Backend API.")
                st.info("Make sure you are running the FastAPI server in another terminal:\n\n`python src/api.py`")
                
        st.info("💡 **How it works:** The Streamlit frontend performs no machine learning! It just sends the image bytes to our FastAPI server and parses the JSON response.")
