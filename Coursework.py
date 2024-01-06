import matplotlib.pyplot as plt
import numpy as np
import keras

# This is to load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalise the pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoded vectors
def one_hot_encoding(labels, num_classes):
    num_samples = len(labels)
    encoded_labels = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        encoded_labels[i, labels[i]] = 1
    return encoded_labels


num_classes = 10
y_train_one_hot = one_hot_encoding(y_train, num_classes)
y_test_one_hot = one_hot_encoding(y_test, num_classes)

# Activation Functions and Their Derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exponent_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exponent_vals / np.sum(exponent_vals, axis=1, keepdims=True)

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (
                np.random.rand(*x.shape) < self.dropout_rate
            ) / self.dropout_rate
            return x * self.mask
        else:
            return x

    def backward(self, grad):
        return grad * self.mask

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation="relu", dropout_rate=0.0):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.layers = []
        self.activation = activation
        self.dropout = Dropout(dropout_rate)
        self.build_network()

    def build_network(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)