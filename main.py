import numpy as np
import matplotlib.pyplot as plt
import copy

from src.dense_layer import DenseLayer
from src.model import Model

def mse(x, y):
    return np.mean((x - y)**2)

def main():

    # create dataset
    x = np.linspace(0, 1, 100, dtype=np.float32)
    y = np.sin(x * 10)

    units = 20
    model = Model(loss='mse', layers=[
        DenseLayer(1, units, 'relu'),
        DenseLayer(units, units, 'relu'),
        DenseLayer(units, 1)
    ])

    batch_size = 10
    num_batches = len(x) // batch_size

    # fitting
    epochs = 10000
    for e in range(epochs):
        model.fit(x, y)

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
