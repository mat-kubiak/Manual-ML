import copy
from abc import ABC, abstractmethod
import numpy as np
from src.activations import ensure_activation
from src.initializers import ensure_initializer

class Layer(ABC):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.params = {}

    @abstractmethod
    def apply_linear(self, tensor):
        pass

    @abstractmethod
    def apply(self, tensor):
        pass

    @abstractmethod
    def backward(self, z_input, prev_act, delta):
        pass

class DenseLayer(Layer):
    def __init__(self, input_shape, units, activation='', initializer=''):
        super().__init__(input_shape, units)
        self.activation = ensure_activation(activation)
        self.initializer = ensure_initializer(initializer)

        self.params = {
            'biases': np.zeros((self.output_shape), dtype=np.float32),
            'weights': self.initializer((self.input_shape, self.output_shape))
        }

    def apply(self, tensor):
        tensor = np.matmul(tensor, self.params['weights'], dtype=np.float32) + self.params['biases']
        return self.activation(tensor)

    def apply_linear(self, tensor):
        return np.matmul(tensor, self.params['weights'], dtype=np.float32) + self.params['biases']

    def backward(self, z_input, prev_act, delta):
        # propagate error to before layer activation
        act_deriv = self.activation.apply_derivative(z_input)
        delta = delta * act_deriv

        b_grad = np.sum(delta, axis=0, keepdims=True)
        w_grad = np.matmul(prev_act.T, delta)

        gradient = {}
        gradient['biases'] = b_grad.astype(np.float32)
        gradient['weights'] = w_grad.astype(np.float32)

        # propagate error to before current layer
        delta = np.matmul(delta, self.params['weights'].T)

        return gradient, delta

    def copy(self):
        return copy.deepcopy(self)

    def add_gaussian(self, stddev):
        self.biases = self.biases + np.random.normal(0.0, stddev, self.biases.shape)
        self.weights = self.weights + np.random.normal(0.0, stddev, self.weights.shape)
        return self
