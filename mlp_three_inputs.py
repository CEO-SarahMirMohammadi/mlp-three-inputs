# Exercise 4.2
# Multiple Layer Perceptron with Three Inputs

from sklearn.neural_network import MLPClassifier


# Dataset with three input features
X = [
    [0., 0., 0.],
    [1., 1., 1.],
    [0., 1., 1.],
    [1., 0., 1.]
]

y = [0, 1, 1, 1]


# Create the MLP classifier
clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)


# Train the model
clf.fit(X, y)


# Make predictions
print("Predictions:")
print(clf.predict([
    [2., 2., 2.],
    [-1., -2., -1.]
]))


# Display the shapes of the weight matrices
print("\nCoefficient shapes:")
print([coef.shape for coef in clf.coefs_])