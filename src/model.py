import numpy as np

from src.layer_stack import LayerStack
from src.losses import get_loss
from src.progress_bar import ProgressBar
from src.optimizers import ensure_optimizer

class Model:
    def __init__(self, layers, loss, optimizer):
        self.stack = LayerStack(layers)
        self.loss = get_loss(loss)
        self.optimizer = ensure_optimizer(optimizer)

    def apply(self, batch):
        return self.stack.apply(batch)

    def fit(self, x, y, batch_size, epochs=1):
        ebar = ProgressBar(range(epochs), total_iters=epochs)
        num_batches = len(x) // batch_size

        # per epoch
        for e in ebar.bar:
            indices = np.random.permutation(len(x))

            # Shuffle both x and y using the same indices
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            # x_shuffled = x
            # y_shuffled = y

            x_batches = np.reshape(x_shuffled, [num_batches, batch_size])
            y_batches = np.reshape(y_shuffled, [num_batches, batch_size])

            losses = []

            bar = ProgressBar(range(len(x_batches)), color='cyan')
            for i in bar.bar:
                x_batch = x_batches[i]
                y_batch = y_batches[i]

                self.stack, loss = self.optimizer.apply(self.stack.copy(), self.loss, x_batch, y_batch)
                bar.update_loss(loss)
                losses.append(loss)
                
                if i % 10000 == 0:
                    ebar.update_loss(np.mean(losses))
