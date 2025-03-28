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

def _backprop_gradient(stack: LayerStack, loss: Loss, x, y_true):
    activations, z_inputs, y_pred = stack.forward_trace(x)

    gradients = []

    # propagate error to before loss
    delta = loss.apply_derivative(y_pred, y_true)
    if len(delta.shape) == 1:
        delta = delta.reshape(-1, 1)

    # propagate through layers
    for i in range(len(stack.layers)-1, -1, -1):
        prev_act = activations[i-1] if i > 0 else x
        gradient, delta = stack.layers[i].backward(z_inputs[i], prev_act, delta)
        gradients.append(gradient)

    return [x for x in reversed(gradients)]

class SGD(Optimizer):
    def __init__(self, lr_rate=0.01, momentum=0.9):
        self.lr_rate = lr_rate
        self.momentum = momentum
        self.velocity = None

    def _update_velocity(self, grad):
        if self.velocity == None:
            self.velocity = grad
            return

        for i in range(len(grad)):
            for k in grad[i].keys():
                self.velocity[i][k] = self.velocity[i][k] * self.momentum + grad[i][k] * (1 - self.momentum)

    def apply(self, stack, loss, batch_x, batch_y):
        gradients = _backprop_gradient(stack, loss, batch_x, batch_y)
        self._update_velocity(gradients)

        for i, layer in enumerate(stack.layers):
            layer.params = {
                k: layer.params[k] - self.lr_rate * self.velocity[i][k]
                for k in layer.params.keys()
            }

        pred = stack.apply(batch_x)
        loss_val = np.mean(loss(pred, batch_y))

        return stack, loss_val

class Adam(Optimizer):
    def __init__(self, lr_rate=0.01, momentum=0.9):
        self.lr_rate = lr_rate
        self.momentum = momentum
        self.velocity = None
        self.second_velocity = None

    def _update_velocity(self, grad):
            if self.velocity == None:
                self.velocity = grad
                self.second_velocity = [{key: np.square(layer[key]) for key in layer.keys()} for layer in grad]
                return

            for i in range(len(grad)):
                for k in grad[i].keys():
                    self.velocity[i][k] = self.velocity[i][k] * self.momentum + grad[i][k] * (1 - self.momentum)
                    self.second_velocity[i][k] = self.second_velocity[i][k] * self.momentum + np.square(grad[i][k]) * (1 - self.momentum)

    def apply(self, stack, loss, batch_x, batch_y):
        gradients = _backprop_gradient(stack, loss, batch_x, batch_y)
        self._update_velocity(gradients)

        epsilon = np.finfo(np.float32).eps

        for i, layer in enumerate(stack.layers):

            for k in layer.params.keys():
                unbiased_momentum_1 = self.velocity[i][k] / (1 - self.momentum)
                unbiased_momentum_2 = self.second_velocity[i][k] / (1 - self.momentum)
                layer.params[k] = layer.params[k] - self.lr_rate * (unbiased_momentum_1) / (np.sqrt(unbiased_momentum_2) + epsilon)

        pred = stack.apply(batch_x)
        loss_val = np.mean(loss(pred, batch_y))
        return stack, loss_val

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
    if not isinstance(a, str):
        raise TypeError(f'Expected an instance of `Optimizer` or `str`, got `{type(a).__name__}`')
    return get_optimizer(opt)
