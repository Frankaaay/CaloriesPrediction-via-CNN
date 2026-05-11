# Calories Prediction via CNN

An industrial-grade deep learning pipeline that uses a Convolutional Neural Network (CNN) to estimate the calories in a food image. Built around the Nutrition5k dataset, this project processes overhead dish imagery and nutrition metadata into a modular PyTorch application.

Dataset reference: [Nutrition5k Dataset](https://github.com/google-research-datasets/Nutrition5k?tab=readme-ov-file)

## Architecture

This application wraps a `ResNet50` architecture. The final fully-connected (FC) layer has been modified into a regression head that maps extracted image features to a single float output (calories).

| Component | Stack | Description |
|-----------|-------|-------------|
| **Data Loader** | `torch.utils.data` / `pandas` | Custom parser that pairs `realsense_overhead/*.png` to `nutrition_data.csv` calorie labels. |
| **Model** | `torchvision.models` | `ResNet50` fine-tuned using AdamW and an L1Loss (MAE) objective. |
| **Pipeline** | Modular Python `.py` | Transitioned from exploratory EDA to script-based training and inference. |

## Project Structure

```text
├── src/
│   ├── data_loader.py         # Custom FoodDataset and transform pipelines
│   ├── model.py               # Pretrained ResNet50 definitions
│   ├── train.py               # Training loop and metric evaluation
│   └── predict.py             # Model inference on single images
├── notebooks/
│   └── 01_eda_and_prototyping.ipynb # Original data exploration notebook
├── tests/
│   └── test_data_loader.py    # Pytest unit tests for tensor transformations
├── data/                      # Local datasets (not tracked)
├── model.pth                  # Saved weights (not tracked)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Quick Start

### 1. Environment Setup

It is highly recommended to use Python 3.12 and a virtual environment.

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare Data

Download the [Nutrition5k imagery archive](https://github.com/google-research-datasets/Nutrition5k) and extract it to the root of the project (e.g. `realsense_overhead/`).

### 3. Run Tests

Verify your data loader logic runs nominally before kicking off expensive training.

```bash
pytest tests/
```

### 4. Train the Model

The default configuration expects a `realsense_overhead/` folder and `nutrition_data.csv` in your root.

```bash
python src/train.py --data_dir ./realsense_overhead/ --metadata ./nutrition_data.csv --epochs 10 --save_path model.pth
```

### 5. Prediction / Inference

Test the model's accuracy on an unseen pizza or salad.

```bash
python src/predict.py realsense_overhead/dish_1556575273.png --model_path model.pth
```
