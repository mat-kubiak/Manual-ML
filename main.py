import numpy as np
import matplotlib.pyplot as plt
import copy

from src.dense_layer import DenseLayer

def mse(x, y):
    return np.mean((x - y)**2)

class LayerStack:
    def __init__(self, layers):
        self.layers = layers
        try:
            for i in range(len(layers)-1):
                self.ensure_layers_match(layers[i], layers[i+1])
        except Exception as e:
            raise Exception(f'Cannot construct layer stack (i={i}): {e}')

    def ensure_layers_match(self, lprev, lnext):
        i_shape = lnext.input_shape
        o_shape = lprev.output_shape

        if i_shape != o_shape:
            raise Exception(f'input shape [{i_shape}] is incompatible with output shape [{o_shape}]')

    def apply(self, batch):
        x = batch.reshape(-1, 1)
        for i in range(len(self.layers)):
            x = self.layers[i].apply(x)
        return x

    def copy(self):
        return copy.deepcopy(self)

    def add_gaussian(self, stddev):
        for layer in self.layers:
            layer.add_gaussian(stddev)
        return self

def fit(stack, batch_x, batch_y):
    init_preds = stack.apply(batch_x)
    init_loss = mse(init_preds.flatten(), batch_y)

    best_stack = stack
    best_loss = init_loss

    tries = 3
    for i in range(tries):
        new_stack = stack.copy().add_gaussian(stddev=0.035)
        new_preds = new_stack.apply(batch_x)
        new_loss = mse(new_preds.flatten(), batch_y)

        if new_loss < best_loss:
            best_stack = new_stack

    print(f'loss: {best_loss}')
    return best_stack

def main():

    # create dataset
    x = np.linspace(0, 1, 100, dtype=np.float32)
    y = np.sin(x * 10)

    units = 20
    model = LayerStack(layers=[
        DenseLayer(1, units, 'relu'),
        DenseLayer(units, units, 'relu'),
        DenseLayer(units, 1)
    ])

    batch_size = 10
    num_batches = len(x) // batch_size

    # fitting
    epochs = 10000
    for e in range(epochs):
        model = fit(model, x, y)

    # evaluating
    preds = np.empty((0,), dtype=np.float32)
    for i in range(num_batches):
        x_batch = x[i*batch_size : (i+1)*batch_size]
        y_batch = y[i*batch_size : (i+1)*batch_size]

        # Apply the neural network on this batch
        x_batch = x_batch.reshape(-1,1)
        predictions = model.apply(x_batch)
        predictions = predictions.flatten()
        preds = np.concatenate([preds, predictions])

        loss = mse(predictions, y_batch)
        print(f'loss: {loss}')

    # visualize results
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    ax.plot(x, preds, linewidth=2.0)
    plt.show()



if __name__ == '__main__':
    main()
