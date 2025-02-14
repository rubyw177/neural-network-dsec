# Handwritten Letter Classification Using Neural Network

## Overview
This project implements a **neural network** for classifying **handwritten letters** from a dataset of **five different alphabet classes**. The model consists of **three hidden layers** using the **tanh activation function** and a **softmax activation** output layer.

## Features
- **Dataset Processing**: Loads and preprocesses the handwritten letter dataset.
- **Neural Network Implementation**: A feedforward neural network with multiple hidden layers.
- **Activation Functions**: Uses **tanh** activation in hidden layers and **softmax** for classification.
- **Performance Evaluation**: Analyzes accuracy using **confusion matrices and classification metrics**.
- **Visualization**: Displays classification results using **Matplotlib and Seaborn**.

## Dependencies
To run this notebook, install the following Python packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
### 1. Import Required Libraries
```python
from nn_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
```

### 2. Load the Dataset
```python
X_train = np.load("dataset_final/X_train.npy").T
y_train = np.load("dataset_final/y_train.npy").T

X_val = np.load("dataset_final/X_val.npy").T
y_val = np.load("dataset_final/y_val.npy").T

X_test = np.load("dataset_final/X_test.npy").T
y_test = np.load("dataset_final/y_test.npy").T
```

### 3. Train the Neural Network
```python
model = NeuralNetwork()
model.train(X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.01)
```

### 4. Evaluate Model Performance
```python
predictions = model.predict(X_test)
accuracy = model.evaluate(predictions, y_test)
print(f"Model Accuracy: {accuracy:.2f}%")
```

### 5. Visualize Classification Results
```python
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, cmap="Blues")
plt.show()
```

## Project Structure
```
📂 project_root/
├── 📂 dataset_dsec/final/        # Final dataset version for training & testing
├── 📂 dataset_final/             # Preprocessed dataset (.npy format)
├── 📜 dataset_generator.py       # Script to process raw data into structured format
├── 📜 handwritten_letter_nn.ipynb # Jupyter Notebook implementing the model
├── 📜 nn_utils.py                # Utility functions for neural network operations
├── 📜 README.md                  # Project documentation
```

## Model Details
- **Input Layer**: Accepts preprocessed images as input.
- **Hidden Layers**: Three layers using **tanh activation**.
- **Output Layer**: Uses **softmax activation** for multi-class classification.
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Stochastic Gradient Descent (SGD).
- **Evaluation Metrics**: Accuracy, confusion matrix.

## Output
The model classifies handwritten letters and evaluates accuracy using confusion matrices and performance visualizations.
