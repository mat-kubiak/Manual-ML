import copy
import numpy as np

def _ensure_layers_match(lprev, lnext):
    i_shape = lnext.input_shape
    o_shape = lprev.output_shape

    if i_shape != o_shape:
        raise Exception(f'input shape [{i_shape}] is incompatible with output shape [{o_shape}]')

class LayerStack:
    def __init__(self, layers):
        self.layers = layers
        try:
            for i in range(len(layers)-1):
                _ensure_layers_match(layers[i], layers[i+1])
        except Exception as e:
            raise Exception(f'Cannot construct layer stack (i={i}): {e}')

    def apply(self, batch):
        x = batch.reshape(-1, 1)
        for i in range(len(self.layers)):
            x = self.layers[i].apply(x)
        return x

    def get_activations(self, batch):
        x = batch.reshape(-1, 1)
        activations = []
        for i in range(len(self.layers)):
            x = self.layers[i].apply(x)
            activations.append(x)
        return activations

    def copy(self):
        return copy.deepcopy(self)

    def add_gaussian(self, stddev):
        for layer in self.layers:
            layer.add_gaussian(stddev)
        return self
