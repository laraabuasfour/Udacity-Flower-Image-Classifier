# Udacity-Flower-Image-Classifier

This project was completed as part of the **AI Programming with Python and TensorFlow** Nanodegree by **Udacity**.

## Project Overview

The goal of this project is to build an image classifier that can recognize different types of flowers using **TensorFlow** and **transfer learning**.

The model is trained on the **Oxford Flowers 102 dataset** and uses a pre-trained **MobileNet** model as a feature extractor.

The project is divided into two parts:

1. **Jupyter Notebook**
   - Load and preprocess the dataset
   - Build and train the classifier
   - Test model performance
   - Save the trained model
   - Make sample predictions

2. **Command-Line Application**
   - Load a saved model
   - Predict the class of an image
   - Display top K predictions
   - Optionally map class labels to flower names using a JSON file

## Technologies Used

- Python
- TensorFlow
- TensorFlow Hub
- NumPy
- Matplotlib
