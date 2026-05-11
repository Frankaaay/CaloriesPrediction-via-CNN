# Calories Prediction via CNN

[![CI Pipeline](https://github.com/Frankaaay/CaloriesPrediction-via-CNN/actions/workflows/ci.yml/badge.svg)](https://github.com/Frankaaay/CaloriesPrediction-via-CNN/actions/workflows/ci.yml)

An industrial-grade deep learning pipeline that uses a Convolutional Neural Network (CNN) to estimate the calories in a food image. Built around the Nutrition5k dataset, this project processes overhead dish imagery and nutrition metadata into a modular PyTorch application.

Dataset reference: [Nutrition5k Dataset](https://github.com/google-research-datasets/Nutrition5k?tab=readme-ov-file)

## Architecture

This application wraps a `ResNet50` architecture. The final fully-connected (FC) layer has been modified into a regression head that maps extracted image features to a single float output (calories).

| Component | Stack | Description |
|-----------|-------|-------------|
| **Data Loader** | `torch.utils.data` / `pandas` | Custom parser that pairs `realsense_overhead/*.png` to `metadata/nutrition_data.csv` calorie labels. |
| **Model** | `torchvision.models` | `ResNet50` fine-tuned using AdamW and an L1Loss (MAE) objective. |
| **Pipeline** | Modular Python `.py` | Transitioned from exploratory EDA to script-based training and inference. |

## Project Structure

```text
├── src/
│   ├── data_loader.py         # Custom FoodDataset and transform pipelines
│   ├── model.py               # Pretrained ResNet50 definitions
│   ├── train.py               # Training loop and metric evaluation
│   ├── evaluate.py            # Generates MAE and RMSE metrics on the test dataset
│   └── predict.py             # Model inference on single images
├── notebooks/
│   └── 01_eda_and_prototyping.ipynb # Original data exploration notebook
├── tests/
│   └── test_data_loader.py    # Pytest unit tests for tensor transformations
├── metadata/                  # Contains dataset CSVs including nutrition_data.csv
├── data/                      # Local datasets (not tracked)
├── model.pth                  # Saved weights (not tracked)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Setup & Running Locally

### 1. Environment Setup

It is highly recommended to use Python 3.12 and a virtual environment.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Unix)
source venv/bin/activate

# Install dependencies (incorporates Streamlit, Torch, gsutil for downloads)
pip install -r requirements.txt
pip install gsutil
```

### 2. Download Image Dataset Locally
Use the `gsutil` package to recursively fetch the dataset from Google Cloud into your root directory:
```bash
gsutil -q -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" .
```

### 3. Local Training & Inference 

**Train:**
```bash
python src/train.py --data_dir realsense_overhead/ --metadata metadata/nutrition_data.csv --epochs 10 --save_path model.pth
```

**Evaluate Accuracy (Requires a saved model.pth):**
```bash
python src/evaluate.py --data_dir realsense_overhead/ --metadata metadata/nutrition_data.csv --model_path model.pth
```

### 4. Running the Microservices (Locally)
If you don't want to use Docker, you can boot the decoupled FastAPI backend and Streamlit frontend manually in two separate terminal windows:

**Terminal 1 (Backend):**
```bash
python src/api.py
```
**Terminal 2 (Frontend):**
```bash
streamlit run app.py
```

### 5. Running with Docker Compose (Production-Ready)
This project is fully containerized. You can boot the entire microservice architecture without installing Python locally. Make sure you have downloaded or trained `model.pth` and placed it in the root directory.

```bash
docker-compose up --build
```
- Frontend available at: `http://localhost:8501`
- Backend API available at: `http://localhost:8000/docs`

## Google Colab Training Workflow

If you want to train this model on Google Colab (recommended for GPU acceleration), follow these exact steps in your Colab notebook:

```python
# 1. Clone your repo
!git clone https://github.com/Frankaaay/CaloriesPrediction-via-CNN
%cd CaloriesPrediction-via-CNN

# 2. Download the Nutrition5k Image Dataset directly to Colab
# Use the same command for local dataset downloading
!gsutil -q -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" .

# 3. Install requirements
!pip install -r requirements.txt

# 4. Train the model!
!python src/train.py --data_dir realsense_overhead/ --metadata metadata/nutrition_data.csv --epochs 20 --save_path model.pth
```
