# Calories Prediction via CNN

This project uses a convolutional neural network to estimate the calories in a food image. The pipeline is built around the Nutrition5k dataset, which provides overhead dish imagery plus metadata for dish and ingredient nutrition values. The repository focuses on turning the raw image archive into a flat training set, pairing each image with calorie labels, and training a ResNet-based regression model.

Dataset reference: https://github.com/google-research-datasets/Nutrition5k?tab=readme-ov-file

## Project Structure

- `src/food_is_good.ipynb` is the expanded notebook version with dataset setup, training, evaluation, and inference examples.
- `nutrition_data.csv` contains the calorie labels used by the dataset loader.
- `metadata/` stores supporting Nutrition5k metadata files.

## Setup

1. Use Python 3.12 or a compatible Python 3 environment.
2. Create and activate a virtual environment.
3. Install the project dependencies:

```bash
pip install -r requirements.txt
```

The notebook and training script also rely on `torch`, `torchvision`, `pandas`, `numpy`, and `Pillow`.

## Data Preparation

1. Download the Nutrition5k overhead imagery archive, for example:

```bash
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" .
```

2. Use the notebook cells to unzip the archive, inspect the extracted files, and prepare the image folder for training.
3. If you want a flat image directory, run the same cleanup logic from the notebook or adapt it into a small helper script.

## Training Workflow

The notebook is now the main workflow and follows this sequence:

1. Unzip or load the image archive.
2. Define a `FoodDataset` that matches image files with calorie labels from `nutrition_data.csv`.
3. Split the dataset into training and test subsets.
4. Fine-tune a pretrained ResNet50 model for calorie regression.
5. Evaluate the model with MAE and RMSE.
6. Run inference on example food images and compare predictions with labels when available.


