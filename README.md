# Multi-Layer Perceptron with Three Inputs

A simple Multi-Layer Perceptron (MLP) classification example using Scikit-Learn's `MLPClassifier`.

This project demonstrates how the input dimensionality of a neural network can be changed while maintaining the same basic MLP architecture.

---

## Overview

A Multi-Layer Perceptron is a feed-forward artificial neural network that can be used for classification and regression tasks.

The original Example 4.2 uses input samples with two features:

```text
[x1, x2]
```

This exercise modifies the program so that every input sample contains three features:

```text
[x1, x2, x3]
```

The model is trained using Scikit-Learn's `MLPClassifier`.

---

## Neural Network Architecture

The model is configured as:

```text
3 Input Features
       │
       ▼
Hidden Layer 1
5 Neurons
       │
       ▼
Hidden Layer 2
2 Neurons
       │
       ▼
Output
Classification
```

The architecture is defined with:

```python
hidden_layer_sizes=(5, 2)
```

The number of input neurons is determined automatically from the number of features in the training data.

Because each sample now contains three values, the input layer has three input features.

---

## Dataset

This project uses a small manually defined dataset.

The input matrix contains four samples, with three features per sample:

```python
X = [
    [0., 0., 0.],
    [1., 1., 1.],
    [0., 1., 1.],
    [1., 0., 1.]
]
```

The corresponding class labels are:

```python
y = [0, 1, 1, 1]
```

Therefore:

* Number of samples: **4**
* Number of input features: **3**
* Number of classes: **2**

This small dataset is intended for educational purposes and demonstrates the structure of a classification problem rather than representing a real-world dataset.

---

## Project Structure

```text
mlp-three-inputs/
│
├── mlp_three_inputs.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

| Package      | Purpose                                          |
| ------------ | ------------------------------------------------ |
| Scikit-Learn | MLP classifier and neural network implementation |

Python 3.9+ is recommended.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/CEO-SarahMirMohammadi/mlp-three-inputs.git
```

Navigate into the project:

```bash
cd mlp-three-inputs
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependency:

```bash
pip install -r requirements.txt
```

---

## Running the Program

Run:

```bash
python mlp_three_inputs.py
```

The program trains the neural network and then produces predictions for two new three-feature input samples.

It also prints the shapes of the learned coefficient matrices.

---

## Implementation

### Three Input Features

The main modification from Example 4.2 is the addition of a third input feature.

Each observation now has the structure:

```text
[x1, x2, x3]
```

For example:

```python
[1., 1., 1.]
```

contains three input values.

### Model Configuration

The classifier is created with:

```python
clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)
```

### Solver

The `lbfgs` solver is used to optimize the neural network parameters.

### Alpha

`alpha=1e-5` controls the strength of L2 regularization.

Regularization can help prevent overly complex model parameters.

### Hidden Layers

```python
hidden_layer_sizes=(5, 2)
```

creates two hidden layers:

* First hidden layer: 5 neurons
* Second hidden layer: 2 neurons

---

## Training

The model is trained with:

```python
clf.fit(X, y)
```

During training, the neural network learns weights connecting:

```text
Input Features
      ↓
Hidden Layer 1
      ↓
Hidden Layer 2
      ↓
Output
```

---

## Prediction

After training, predictions are made using:

```python
clf.predict([
    [2., 2., 2.],
    [-1., -2., -1.]
])
```

Both samples contain exactly three input features, matching the dimensionality of the training data.

---

## Coefficient Shapes

The program also displays:

```python
[coef.shape for coef in clf.coefs_]
```

`clf.coefs_` contains the learned weight matrices between consecutive layers.

Because the network has three input features and hidden layers of sizes 5 and 2, the first weight matrix has a shape corresponding to:

```text
3 × 5
```

The second corresponds to:

```text
5 × 2
```

The final matrix connects the second hidden layer to the output layer.

This provides a simple way to inspect how the network's architecture is represented internally.

---

## Key Concepts

### Multi-Layer Perceptron

An MLP is a feed-forward neural network consisting of an input layer, one or more hidden layers, and an output layer.

### Input Features

Each input sample is represented by a vector of numerical features.

In this exercise:

```text
3 features per sample
```

### Hidden Layers

Hidden layers transform the input features through learned weights and activation functions.

### Classification

The network learns to map input samples to discrete class labels.

### Neural Network Weights

Weights determine how strongly information is passed between neurons.

The weights are learned during training.

---

## Workflow

```text
Define Three-Feature Dataset
          │
          ▼
Define Class Labels
          │
          ▼
Create MLPClassifier
          │
          ▼
Train Neural Network
          │
          ▼
Predict New Samples
          │
          ▼
Inspect Weight Matrices
```

---

## Expected Output

The exact prediction output can depend on the installed Scikit-Learn version, but the program produces output in this form:

```text
Predictions:
[...]

Coefficient shapes:
[(3, 5), (5, 2), (2, 1)]
```

The important architectural change is that the first coefficient matrix now starts with **3 input features** instead of 2.

---

## Limitations

This example uses only four training samples, which is far too small for a meaningful real-world neural network application.

The purpose of the project is to demonstrate:

* MLP architecture
* Three-dimensional input
* Model training
* Prediction
* Neural network coefficient shapes

It should not be interpreted as a production classification model.

---

## Future Improvements

Possible extensions include:

* Use a larger dataset
* Add more training samples
* Normalize or standardize input features
* Experiment with different hidden-layer sizes
* Compare different activation functions
* Compare different optimization algorithms
* Evaluate the model using accuracy and confusion matrices
* Experiment with different regularization values
* Visualize training performance

---

## Educational Purpose

This repository is an implementation of **Exercise 4.2**, based on Example 4.2 — Multiple Layer Perceptron.

The exercise demonstrates how to modify an MLP classification program from two input features to three input features while keeping the rest of the neural network structure largely unchanged.

---

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute the code for educational and other purposes, subject to the terms of the license.
