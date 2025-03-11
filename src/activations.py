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
        exp = np.exp(x)
        return exp / np.sum(exp)

    def apply_derivative(self, x):
        raise NotImplementedError('derivative of softmax not implemented!')

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
