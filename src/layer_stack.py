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
            raise Exception(f'Cannot construct layer stack: {e}')

    def apply(self, batch):
        x = batch
        for i in range(len(self.layers)):
            x = self.layers[i].apply(x)
        return x

    def forward_trace(self, batch):
        x = batch

        activations = []
        z_inputs = []

        for i in range(len(self.layers)):
            x = self.layers[i].apply_linear(x)
            z_inputs.append(x)
            x = self.layers[i].activation.apply(x)
            activations.append(x)

        y_pred = activations[-1]
        return activations, z_inputs, y_pred

    def copy(self):
        return copy.deepcopy(self)

    def add_gaussian(self, stddev):
        for layer in self.layers:
            layer.add_gaussian(stddev)
        return self
