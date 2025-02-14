# Handwritten Letter Classification Using Neural Network

## Overview
This project implements a neural network for classifying handwritten letters from a dataset of five different alphabet classes. The model consists of three hidden layers using the tanh activation function and a softmax activation output layer.

## Features
- Loads and preprocesses handwritten letter dataset
- Implements a feedforward neural network using multiple hidden layers
- Uses tanh activation in hidden layers and softmax for classification
- Evaluates performance using confusion matrices and accuracy metrics
- Visualizes predictions and performance using matplotlib and seaborn

## Dependencies
To run this notebook, install the following Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
1. Import the required libraries:
   ```python
   from nn_utils import *
   import pandas as pd
   import matplotlib.pyplot as plt
   import random
   from sklearn.metrics import confusion_matrix
   import seaborn as sns
   ```
2. Load the dataset:
   ```python
   X_train = np.load("dataset_final/X_train.npy").T
   y_train = np.load("dataset_final/y_train.npy").T

   X_val = np.load("dataset_final/X_val.npy").T
   y_val = np.load("dataset_final/y_val.npy").T

   X_test = np.load("dataset_final/X_test.npy").T
   y_test = np.load("dataset_final/y_test.npy").T
   ```
3. Train the neural network model with defined layers and activation functions.
4. Evaluate performance using accuracy metrics and confusion matrices.
5. Visualize classification results.

## Repository Structure
### Folders:
- **`dataset_dsec/final`**: Contains the final version of the dataset used for training and evaluation.
- **`dataset_final`**: Stores the preprocessed dataset in `.npy` format, including training, validation, and test splits.

### Files:
- **`README.md`**: Provides an overview of the project, including details about the neural network, dataset, dependencies, and instructions for running the model.
- **`dataset_generator.py`**: Script that processes raw data and converts it into a structured dataset.
- **`handwritten_letter_nn.ipynb`**: Jupyter Notebook containing the implementation of the neural network.
- **`nn_utils.py`**: Utility script containing helper functions for the neural network.

## Output
The model classifies handwritten letters and provides accuracy evaluation using confusion matrices and visualizations.
