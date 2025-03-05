import numpy as np

def mse(x, y):
    return np.mean((x - y)**2)

def mae(x, y):
    return np.mean(np.absolute(x - y))

def get_loss(name):
    fns = {
        'mse': mse,
        'mae': mae,
    }
    if name not in fns.keys():
        raise Exception(f'activation function `{name}` not found! Available activation functions: [{', '.join(fns.keys())}]')
    return fns[name]
