import numpy as np

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def get_activation(name):
    fns = {
        'linear': linear,
        'relu': relu,
    }

    if name == '':
        name = 'linear'
    if name not in fns.keys():
        raise Exception(f'Activation function `{name}` not found! Available activations: [{', '.join(fns.keys())}]')
    return fns[name]
