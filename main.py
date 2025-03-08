import numpy as np
import matplotlib.pyplot as plt
import copy

from src.dense_layer import DenseLayer
from src.model import Model
from src import optimizers

def main():

    # create dataset
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    y = np.sin(x * 10)

    units = 50
    model = Model(
        loss='mse',
        optimizer=optimizers.SGD(lr_rate=0.0002),
        layers=[
            DenseLayer(1, units, 'tanh'),
            DenseLayer(units, units, 'tanh'),
            DenseLayer(units, units, 'tanh'),
            DenseLayer(units, 1)
        ]
    )

    batch_size = 100
    num_batches = len(x) // batch_size
    loss_history = model.fit(x, y, batch_size=batch_size, epochs=700)

    # evaluating
    preds = np.empty((0,), dtype=np.float32)
    loss = 0.0
    for i in range(num_batches):
        x_batch = x[i*batch_size : (i+1)*batch_size]
        y_batch = y[i*batch_size : (i+1)*batch_size]

        # Apply the neural network on this batch
        x_batch = x_batch.reshape(-1,1)
        predictions = model.apply(x_batch)
        predictions = predictions.flatten()
        preds = np.concatenate([preds, predictions])

        loss += model.loss(predictions, y_batch)

    loss /= num_batches
    print(f'\nFinal loss: {loss}')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # True labels vs. Predictions
    axes[0].plot(x, y, label="True Labels", linewidth=2.0)
    axes[0].plot(x, preds, label="Predictions", linewidth=2.0)
    axes[0].set_title("True Labels vs. Predictions")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].legend()
    axes[0].grid()

    # Loss History
    axes[1].plot(range(len(loss_history)), loss_history, color='red', linewidth=2.0)
    axes[1].set_title("Loss History")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
