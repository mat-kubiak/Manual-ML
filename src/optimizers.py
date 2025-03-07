import numpy as np
from abc import ABC, abstractmethod
from src.layer_stack import LayerStack
from src.losses import Loss

class Optimizer(ABC):
    @abstractmethod
    def apply(self, stack: LayerStack, loss: Loss, batch_x, batch_y):
        pass

    def get_name(self):
        return type(self).__name__.lower()

class RandomOptimizer(Optimizer):
    def apply(self, stack, loss, batch_x, batch_y):
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

    def get_name(self):
        return 'random_optimizer'

class SGD(Optimizer):
    def __init__(self, lr_rate=0.01):
        self.lr_rate = lr_rate

    def apply(self, stack, loss, batch_x, batch_y):
        bias_gradients, weight_gradients = self._backprop_gradient(stack, loss, batch_x, batch_y)

        for i, layer in enumerate(stack.layers):
            layer.biases = layer.biases - self.lr_rate * bias_gradients[i]
            layer.weights = layer.weights - self.lr_rate * weight_gradients[i]

        pred = stack.apply(batch_x)
        loss_val = loss(batch_y, pred)

        return stack, loss_val

    def _backprop_gradient(self, stack: LayerStack, loss:Loss, x, y_true):
        activations = stack.get_activations(x)
        y_pred = activations[len(activations)-1].flatten()

        bias_gradients = []
        weight_gradients = []

        # initialize delta
        delta = loss.apply_derivative(y_pred, y_true).reshape(-1, 1)

        for i in range(len(stack.layers)-1, -1, -1):
            prev_act = activations[i-1] if i != 0 else x.reshape(-1,1)

            current_layer = stack.layers[i]
            prev_layer = stack.layers[i-1]
            weights = current_layer.weights

            act_deriv = current_layer.activation.apply_derivative(prev_act)
            b_grad = np.sum(delta * act_deriv, axis=0, keepdims=False)

            w_grad = np.matmul(prev_act.T, delta)

            bias_gradients.append(b_grad)
            weight_gradients.append(w_grad)

            if i != 0:
                delta = np.matmul(delta, weights.T)

        return list(reversed(bias_gradients)), list(reversed(weight_gradients))



_optimizers = {
    cls().get_name(): cls
    for cls in Optimizer.__subclasses__()
}

def get_optimizer(name):
    name = name.strip().lower()
    if name not in _optimizers.keys():
        available = ', '.join(_losses.keys())
        raise ValueError(f'Optimizer `{name}` not found! Available optimizers: [{available}]')
    return _optimizers[name]()

def ensure_optimizer(opt):
    if isinstance(opt, Optimizer):
        return opt

    return get_optimizer(opt)
