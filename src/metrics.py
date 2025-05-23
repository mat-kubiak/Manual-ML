import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self):
        self.value = 0
        self.counter = 0

    @abstractmethod
    def _update(self, y_pred, y_true):
        pass

    def update(self, y_pred, y_true):
        self._update(y_pred, y_true)
        self.counter += 1

    def get(self):
        return self.value / self.counter

    def reset(self):
        self.value = 0
        self.counter = 0

    def get_name(self):
        return type(self).__name__.lower()

class MSE(Metric):
    def _update(self, y_pred, y_true):
        self.value += np.mean((y_pred - y_true)**2)

class MAE(Metric):
    def _update(self, y_pred, y_true):
        self.value += np.mean(np.absolute(y_pred - y_true))

class Accuracy(Metric):
    def _update(self, y_pred, y_true):
        if y_pred.ndim != 2 or y_pred.shape[1] == 1:
            raise Exception(f'Accuracy metric works only on tensors with shape: (batch_size, classes>1). Got shape: {y_pred.shape}')

        x = np.argmax(y_pred, axis=1)
        y = np.argmax(y_true, axis=1)

        self.value += np.mean(x == y)

_metrics = {
    cls().get_name(): cls
    for cls in Metric.__subclasses__()
}

def get_metric(name):
    name = name.strip().lower()
    if name not in _metrics.keys():
        available = ', '.join(_metrics.keys())
        raise ValueError(f'Metric `{name}` not found! Available metrics: [{available}]')
    return _metrics[name]()

def ensure_metric(m):
    if isinstance(m, Metric):
        return m
    if not isinstance(m, str):
        raise TypeError(f'Expected an instance of `Metric` or `str`, got `{type(a).__name__}`')
    return get_metric(m)
