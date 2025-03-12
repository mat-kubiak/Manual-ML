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

    def fit(self, x, y, batch_size, epochs, epoch_callback=None):
        loss_history = []
        total_duration = 0.0

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f'x and y should have the same number of samples, found: `{x.shape[0]}` and `{y.shape[0]}`')

        num_samples = x.shape[0]

        odd_batch_size = num_samples % batch_size
        is_even = odd_batch_size == 0

        num_even_batches = num_samples // batch_size
        num_batches_total = num_even_batches + int(not is_even)

        for e in range(epochs):
            # Shuffling
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            if not is_even:
                x_odd = x_shuffled[-odd_batch_size:]
                y_odd = y_shuffled[-odd_batch_size:]

                x_shuffled = x_shuffled[0:num_samples-odd_batch_size]
                y_shuffled = y_shuffled[0:num_samples-odd_batch_size]

            x_batches = np.reshape(x_shuffled, [num_even_batches, batch_size] + list(x.shape[1:]))
            y_batches = np.reshape(y_shuffled, [num_even_batches, batch_size] + list(y.shape[1:]))

            if not is_even:
                x_batches = list(x_batches) +  list([x_odd])
                y_batches = list(y_batches) + list([y_odd])

            print(f'\nEpoch [{e+1}/{epochs}]:')

            epoch_loss = 0.0
            bbar = ProgressBar(total_iters=num_batches_total)
            for i in range(num_batches_total):
                x_batch = x_batches[i]
                y_batch = y_batches[i]

                self.stack, loss = self.optimizer.apply(self.stack, self.loss, x_batch, y_batch)
                epoch_loss += loss
                bbar.update(epoch_loss / (i+1))

            duration = bbar.close()
            total_duration += duration

            loss_history.append(epoch_loss / num_batches_total)
            if epoch_callback != None:
                epoch_callback.__call__(e+1)

        stats = {
            'total_duration': total_duration,
            'mean_epoch_duration': total_duration/epochs,
            'loss_history': loss_history,
        }

        return stats
