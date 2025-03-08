import copy
import numpy as np
from src.activations import ensure_activation

class DenseLayer:
    def __init__(self, input_shape, units, activation=''):
        self.input_shape = input_shape
        self.output_shape = units
        self.activation = ensure_activation(activation)

        stddev = 0.2
        self.biases = np.random.normal(0.0, stddev, [units]).astype(np.float32)
        self.weights = np.random.normal(0.0, stddev, [input_shape, units]).astype(np.float32)

    def apply(self, tensor):
        tensor = np.matmul(tensor, self.weights) + self.biases
        tensor = self.activation(tensor)
        return tensor

    def apply_linear(self, tensor):
        return np.matmul(tensor, self.weights) + self.biases

    def copy(self):
        return copy.deepcopy(self)

    def add_gaussian(self, stddev):
        self.biases = self.biases + np.random.normal(0.0, stddev, self.biases.shape)
        self.weights = self.weights + np.random.normal(0.0, stddev, self.weights.shape)
        return self
