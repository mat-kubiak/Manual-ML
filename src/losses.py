import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def apply(self, x, y):
        pass

    def __call__(self, x, y):
        return self.apply(x, y)

    def get_name(self):
        return type(self).__name__.lower()

class MSE(Loss):
    def apply(self, x, y):
        return np.mean((x - y)**2)

class MAE(Loss):
    def apply(self, x, y):
        return np.mean(np.absolute(x - y))

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
