import numpy as np
from tqdm import tqdm

from src.layer_stack import LayerStack
from src.losses import get_loss

def naive_optimizer(stack, loss, batch_x, batch_y):
    init_preds = stack.apply(batch_x)
    init_loss = loss(init_preds.flatten(), batch_y)

    best_stack = stack
    best_loss = init_loss

    tries = 3
    for i in range(tries):
        new_stack = stack.copy().add_gaussian(stddev=0.035)
        new_preds = new_stack.apply(batch_x)
        new_loss = loss(new_preds.flatten(), batch_y)

        if new_loss < best_loss:
            best_stack = new_stack

    return best_stack, best_loss

class Model:
    def __init__(self, layers, loss):
        self.stack = LayerStack(layers)
        self.loss_fn = get_loss(loss)
        self.optimizer_fn = naive_optimizer

    def apply(self, batch):
        return self.stack.apply(batch)

    def fit(self, batch_x, batch_y, epochs=1):
        bar_format="\033[92m{bar:30}\033[0m | epoch {n_fmt}/{total_fmt} ({percentage:.1f}%) | {desc}"

        for e in (pbar := tqdm(range(epochs), bar_format=bar_format, ascii="─━")):
            self.stack, loss = self.optimizer_fn(self.stack, self.loss_fn, batch_x, batch_y)
            pbar.set_description_str(f"loss: {loss:.5f} ")
