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

class Siren(Initializer):
    def __init__(self, is_first=False, omega_0=30.0):
        self.is_first = is_first
        self.omega_0 = omega_0

    def __call__(self, shape):
        in_features = shape[0]
        if self.is_first:
            return np.random.uniform(
                -1.0 / in_features,
                1.0 / in_features,
                size=shape
            )
        else:
            return np.random.uniform(
                -np.sqrt(6 / in_features) / self.omega_0,
                np.sqrt(6 / in_features) / self.omega_0,
                size=shape
            )

    def get_name():
        return 'siren'

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

