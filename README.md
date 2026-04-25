# UCIrvine-CS172B-CaloriesPrediction-via-CNN

This project aims to help people in their day-to-day lives. Based on an image of a dish, our Neural Network predicts the calories in that image. We are using an open-source dataset that contains videos of food as well as nutrients and calories for single ingredients and entire dishes. By using a Convolutional Neural Network, we want to get features such as what dish we're looking at, what ingredients are contained, how greasy the food looks etc. and then perform regression to predict calories.


https://github.com/google-research-datasets/Nutrition5k?tab=readme-ov-file

# Walkthrough
* Ideally, Python 3.12 is installed
* Run ```source venv/activate/bin```
* Run ```pip3 install -r requirements.txt```
* Download training data: ```gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" .```
* Run processing.py to resize, rename and move data to a single directory
* Run ```find /path/to/172B-food-project/realsense_overhead -mindepth 1 -type d -exec rm -r {} +``` to delete the remaining empty directories
* Use the ipynb instead of the train.py
