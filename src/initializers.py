import numpy as np
from abc import ABC, abstractmethod

class Initializer(ABC):
    @abstractmethod
    def __call__(self, shape):
        pass

    @staticmethod
    def get_name():
        pass


class Uniform(Initializer):
    def __init__(self, low=-0.5, high=0.5):
        self.low = low
        self.high = high

    def __call__(self, shape):
        return np.random.uniform(self.low, self.high, size=shape)

    def get_name():
        return 'uniform'

_initializers = {
    cls.get_name(): cls
    for cls in Initializer.__subclasses__()
}

def get_initializer(name):
    name = name.strip().lower() or 'uniform'
    if name not in _initializers.keys():
        available = ', '.join(_initializers.keys())
        raise ValueError(f'Initializer `{name}` not found! Available initializers: [{available}]')
    return _initializers[name]()

def ensure_initializer(l):
    if isinstance(l, Initializer):
        return l
    if not isinstance(l, str):
        raise TypeError(f'Expected an instance of `Initializer` or `str`, got `{type(a).__name__}`')
    return get_initializer(l)

