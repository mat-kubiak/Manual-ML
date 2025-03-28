import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def apply(self, x):
        pass

    @abstractmethod
    def apply_derivative(self, x):
        pass

    def __call__(self, x):
        return self.apply(x)

    def get_name(self):
        return type(self).__name__.lower()


class Linear(Activation):
    def apply(self, x):
        return x

    def apply_derivative(self, x):
        return np.ones_like(x)

class ReLU(Activation):
    def apply(self, x):
        return np.maximum(0, x)

    def apply_derivative(self, x):
        return np.where(x > 0, 1.0, 0.0).astype(np.float32)

class LeakyReLU(Activation):
    def __init__(self, slope=0.01):
        self.slope = slope

    def get_name(self):
        return 'leaky_relu'

    def apply(self, x):
        return np.where(x > 0, x, self.slope * x)

    def apply_derivative(self, x):
        return np.where(x > 0, 1.0, -self.slope).astype(np.float32)

class Sigmoid(Activation):
    def apply(self, x):
        return 1 / (1 + np.exp(-x))

    def apply_derivative(self, x):
        sig = self.apply(x)
        return sig * (1 - sig)

class Tanh(Activation):
    def apply(self, x):
        return np.tanh(x)
    
    def apply_derivative(self, x):
        return 1 - np.tanh(x) ** 2

class Softmax(Activation):
    def apply(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - x_max) # subtract x_max for overflow prevention
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def apply_derivative(self, x):
        softmax_out = self.apply(x)

        batch_size, num_classes = softmax_out.shape
        jacobian = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            s = softmax_out[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(s) - np.dot(s, s.T)  # diag(S) - S * S^

        return jacobian

class Sine(Activation):
    def __init__(self, freq=30.0, amp=1.0):
        self.f = freq
        self.a = amp

    def apply(self, x):
        return np.sin(x * self.f) * self.a

    def apply_derivative(self, x):
        return self.f * self.a * np.cos(x * self.f)

_activations = {
    cls().get_name(): cls
    for cls in Activation.__subclasses__()
}

def get_activation(name):
    name = name.strip().lower() or 'linear'
    if name not in _activations.keys():
        available = ', '.join(_activations.keys())
        raise ValueError(f'Activation function `{name}` not found! Available activations: [{available}]')
    return _activations[name]()

def ensure_activation(a):
    if isinstance(a, Activation):
        return a
    if not isinstance(a, str):
        raise TypeError(f'Expected an instance of `Activation` or `str`, got `{type(a).__name__}`')
    return get_activation(a)
