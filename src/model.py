import numpy as np

from src.layer_stack import LayerStack
from src.losses import ensure_loss
from src.progress_bar import ProgressBar
from src.optimizers import ensure_optimizer

class Model:
    def __init__(self, layers, loss, optimizer):
        self.stack = LayerStack(layers)
        self.loss = ensure_loss(loss)
        self.optimizer = ensure_optimizer(optimizer)

    def apply(self, batch):
        return self.stack.apply(batch)

    def fit(self, x, y, batch_size, epochs=1):
        loss_history = []

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f'x and y should have the same number of samples, found: `{x.shape[0]}` and `{y.shape[0]}`')

        num_samples = x.shape[0]
        num_batches = num_samples // batch_size

        ebar = ProgressBar(range(epochs), total_iters=epochs)
        for e in ebar.bar:
            # Shuffling
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            x_batches = np.reshape(x_shuffled, [num_batches, batch_size] + list(x.shape[1:]))
            y_batches = np.reshape(y_shuffled, [num_batches, batch_size] + list(y.shape[1:]))

            epoch_loss = 0.0
            for i in range(num_batches):
                x_batch = x_batches[i]
                y_batch = y_batches[i]

                self.stack, loss = self.optimizer.apply(self.stack.copy(), self.loss, x_batch, y_batch)
                epoch_loss += loss

            ebar.update_loss(epoch_loss / num_batches)
            loss_history.append(epoch_loss / num_batches)

        return loss_history
