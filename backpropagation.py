import numpy as np
import pandas as pd

def one_hot_encode_columns(matrix):
    encoded_columns = []
    num_columns = matrix.shape[1]
    for col_idx in range(num_columns):
        column = matrix[:, col_idx]
        unique_vals = sorted(set(column))
        mapping = {val: i for i, val in enumerate(unique_vals)}
        one_hot = np.zeros((len(column), len(unique_vals)))
        for row_idx, val in enumerate(column):
            one_hot[row_idx, mapping[val]] = 1
        encoded_columns.append(one_hot)
    combined = np.hstack(encoded_columns)
    return combined.T

def apply_weight_updates(layers, updates):
    updated_layers = []
    for w, delta in zip(layers, updates):
        updated_layers.append(w - delta)
    return updated_layers

def classify(data, layers):
    return feedforward(data, layers)[-1]

def activation(x):
    return 1 / (1 + np.exp(-x))

def compute_weight_updates(layers, data, activations, grads, lr):
    inputs = [data] + activations[:-1]
    updates = []
    for w, inp, grad in zip(layers, inputs, grads):
        inp = add_bias(inp)
        delta = inp.dot(grad.T) * lr
        updates.append(delta)
    return updates

def feedforward(data, layers):
    current = data
    activations = []
    for w in layers:
        current = add_bias(current)
        z = np.dot(w.T, current)
        a = activation(z)
        current = a
        activations.append(a)
    return activations

def backpropagate(layers, outputs, targets):
    grads = []
    error = outputs[-1] - targets
    for w, out in zip(reversed(layers), reversed(outputs)):
        grad = activation_derivative(out) * error
        grads.append(grad)
        error = np.dot(w, grad)
        error = np.delete(error, 0, axis=0)
    return list(reversed(grads))

def activation_derivative(x):
    sig = activation(x)
    return sig * (1 - sig)

def mean_squared_error(predicted, actual):
    return np.sum((predicted - actual) ** 2) / len(predicted)

def test_model(layers, data):
    output = classify(data, layers)
    predicted = np.argmax(output, axis=0)
    for i in range(len(predicted)):
        print(f"Row {i}: {predicted[i]}")

def build_model(input_dim, output_dim, hidden_dim):
    layers = []
    sizes = [hidden_dim, output_dim]
    for neurons in sizes:
        weights = np.random.rand(input_dim + 1, neurons) * 2 - 1
        input_dim = neurons
        layers.append(weights)
    return layers

def add_bias(inputs):
    bias = np.ones(inputs.shape[1]).reshape(1, -1)
    return np.vstack([bias, inputs])

def train(layers, data, targets, lr, steps):
    for _ in range(steps):
        activations = feedforward(data, layers)
        grads = backpropagate(layers, activations, targets)
        updates = compute_weight_updates(layers, data, activations, grads, lr)
        layers = apply_weight_updates(layers, updates)
    return layers

dataset = pd.read_csv("Transformed Data Set - Sheet1.csv")
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

inputs = one_hot_encode_columns(features)
class_names, class_indices = np.unique(labels, return_inverse=True)
targets = np.eye(len(class_names))[class_indices].T

for i in range(len(class_names)):
    print(f"Class {i}: {class_names[i]}")

print("----------")

model = build_model(inputs.shape[0], targets.shape[0], 15)
trained_model = train(model, inputs, targets, lr=0.002, steps=500)

test_model(trained_model, inputs)
