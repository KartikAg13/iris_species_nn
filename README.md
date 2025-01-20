# Iris Species Neural Network

This repository contains the implementation of a feedforward neural network for classifying Iris flower species. The neural network is trained using the Adam optimizer and achieves an accuracy of 95.33% on the training dataset.

## Overview
The Iris dataset is a well-known dataset used in machine learning for classification tasks. It consists of 150 samples of Iris flowers, with 50 samples each from three species:
- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

Each sample includes four features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The neural network is implemented from scratch in Python using libraries such as NumPy, Pandas, and Matplotlib.

## Dataset
- **Source**: [Kaggle Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

## Implementation Details
### Data Preprocessing
1. **Mapping Target Labels**: The species labels are mapped to integers:
   - `Iris-setosa`: 0
   - `Iris-versicolor`: 1
   - `Iris-virginica`: 2
2. **One-Hot Encoding**: The target labels are converted to one-hot encoded vectors.
3. **Feature Selection**: Features include sepal length, sepal width, petal length, and petal width.

### Neural Network Architecture
- Input layer: 4 neurons (corresponding to the 4 features).
- Hidden layer 1: 4 neurons with ReLU activation.
- Hidden layer 2: 5 neurons with ReLU activation.
- Output layer: 3 neurons (corresponding to the 3 classes) with softmax activation.

### Training Details
- **Optimizer**: Adam optimizer
- **Learning Rate**: 0.001
- **Epochs**: 1000
- **Batch Size**: Full-batch
- **Loss Function**: Categorical cross-entropy

### Results
- **Training Accuracy**: 95.33%

## Visualizations
1. **Weights and Biases**:
   - Final weights and biases of the output layer are plotted for analysis.
2. **Cost Curve**:
   - The cost function is plotted over epochs to observe convergence.

## Functions
### Core Functions
- `one_hot_encoding`: Converts target labels into one-hot encoded vectors.
- `initialize_parameters`: Initializes weights and biases using He initialization.
- `relu` and `softmax`: Activation functions.
- `forward_propagation`: Computes activations for each layer.
- `compute_cost`: Calculates the loss using categorical cross-entropy.
- `backward_propagation`: Computes gradients for weights and biases.
- `update_parameters`: Updates parameters using the Adam optimizer.
- `train`: Implements the training loop.

### Visualization Functions
- `plot_weights_and_biases`: Plots weights and biases of the output layer.
- `plot_cost_curve`: Plots the cost function over training epochs.

### Evaluation
- `evaluate_model`: Evaluates the model on the training data and calculates accuracy.

## Requirements
- Python 3.7+
- Required Libraries:
  - NumPy
  - Pandas
  - Matplotlib

## File Structure
- `main.ipynb`: Main script for implementing and training the neural network.
- `README.md`: Documentation for the project.

## Future Enhancements
- Implement dropout for regularization.
- Test on a validation/test split to evaluate generalization.
- Add hyperparameter tuning options.
