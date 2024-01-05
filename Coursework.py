import numpy as np
import keras

# This is to load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
