import numpy as np
import tensorflow as tf
from tf.keras.datasets import mnist

# This is to load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
