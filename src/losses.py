import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def apply(self, y_pred, y_true):
        pass

    @abstractmethod
    def apply_derivative(self, y_pred, y_true):
        pass

    def __call__(self, y_pred, y_true):
        return self.apply(y_pred, y_true)

    def get_name(self):
        return type(self).__name__.lower()

class MSE(Loss):
    def apply(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def apply_derivative(self, y_pred, y_true):
        return 2.0*(y_pred - y_true)

class MAE(Loss):
    def apply(self, y_pred, y_true):
        return np.mean(np.absolute(y_pred - y_true))
    
    def apply_derivative(self, y_pred, y_true):
        return np.where(y_pred > y_true, 1, np.where(y_pred < y_true, -1, 0))

class CategoricalCrossentropy(Loss):
    def apply(self, y_pred, y_true):
        epsilon = np.finfo(np.float32).eps
        return -np.sum(y_true * np.log(y_pred + epsilon), axis=1)

    def apply_derivative(self, y_pred, y_true):
        return y_pred - y_true

    def get_name(self):
        return 'categorical_crossentropy'

_losses = {
    cls().get_name(): cls
    for cls in Loss.__subclasses__()
}

def get_loss(name):
    name = name.strip().lower()
    if name not in _losses.keys():
        available = ', '.join(_losses.keys())
        raise ValueError(f'Loss function `{name}` not found! Available losses: [{available}]')
    return _losses[name]()

def ensure_loss(l):
    if isinstance(l, Loss):
        return l
    if not isinstance(l, str):
        raise TypeError(f'Expected an instance of `Loss` or `str`, got `{type(a).__name__}`')
    return get_loss(l)
