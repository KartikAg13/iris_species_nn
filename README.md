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

Install dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/KartikAg13/iris_species_nn.git
   ```
2. Navigate to the project directory:
   ```bash
   cd iris_species_nn
   ```
3. Ensure the Iris dataset (`Iris.csv`) is placed in the same directory.
4. Run the Python script:
   ```bash
   python iris_nn.py
   ```

## File Structure
- `iris_nn.py`: Main script for implementing and training the neural network.
- `Iris.csv`: Dataset file (download from [Kaggle](https://www.kaggle.com/datasets/uciml/iris)).
- `readme.md`: Documentation for the project.

## Future Enhancements
- Implement dropout for regularization.
- Test on a validation/test split to evaluate generalization.
- Add hyperparameter tuning options.

## References
- [He Initialization](https://arxiv.org/abs/1502.01852)
- [Adam Optimizer](https://arxiv.org/abs/1412.6980)
- [Kaggle Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

---
For any questions or suggestions, feel free to open an issue or reach out. Happy coding!

