import copy
import numpy as np
from src.activations import ensure_activation, is_activation_linear

class DenseLayer:
    def __init__(self, input_shape, units, activation=''):
        self.input_shape = input_shape
        self.output_shape = units
        self.activation = ensure_activation(activation)
        self.is_activation_linear = is_activation_linear(self.activation)

        stddev = 0.2
        self.biases = np.random.normal(0.0, stddev, [units]).astype(np.float32)
        self.weights = np.random.normal(0.0, stddev, [input_shape, units]).astype(np.float32)

    def apply(self, tensor):
        tensor = np.matmul(tensor, self.weights) + self.biases
        if self.activation is not None:
            tensor = self.activation(tensor)
        return tensor

    def copy(self):
        return copy.deepcopy(self)

    def add_gaussian(self, stddev):
        self.biases = self.biases + np.random.normal(0.0, stddev, self.biases.shape)
        self.weights = self.weights + np.random.normal(0.0, stddev, self.weights.shape)
        return self
