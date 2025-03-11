import numpy as np
import matplotlib.pyplot as plt

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

    # init graphs
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    true_line, = axes[0].plot(x, y, label="True Labels", linewidth=2.0)
    pred_line, = axes[0].plot(x, np.zeros_like(x), label="Predictions", linewidth=2.0)
    axes[0].set_title("True Labels vs. Predictions")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].legend()
    axes[0].grid()

    loss_line, = axes[1].plot([], [], color='red', linewidth=2.0)
    axes[1].set_title("Loss History")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale('log')
    axes[1].grid()

    plt.tight_layout()
    plt.show(block=False)

    loss_history = []

    def update_plot_callback(epoch):
        preds = model.apply(x.reshape(-1,1)).flatten()
        pred_line.set_ydata(preds)

        loss_history.append(model.loss(preds, y))

        loss_line.set_xdata(range(len(loss_history)))
        loss_line.set_ydata(loss_history)

        axes[1].set_xlim(0, len(loss_history))
        axes[1].relim()
        axes[1].autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

    # training
    loss_history = model.fit(x, y,
        batch_size=100,
        epochs=700,
        epoch_callback=update_plot_callback
    )

    # evaluating
    predictions = model.apply(x.reshape(-1,1)).flatten()
    print(f'\nFinal loss: {model.loss(predictions, y)}')

    # final show
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
