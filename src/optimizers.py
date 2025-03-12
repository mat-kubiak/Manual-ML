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

    bias_gradients = []
    weight_gradients = []

    # propagate error to before loss
    delta = loss.apply_derivative(y_pred, y_true)
    if len(delta.shape) == 1:
        delta = delta.reshape(-1, 1)

    for i in range(len(stack.layers)-1, -1, -1):
        # propagate error to before layer activation
        act_deriv = stack.layers[i].activation.apply_derivative(z_inputs[i])
        delta = delta * act_deriv

        prev_act = activations[i-1] if i > 0 else x
        
        b_grad = np.sum(delta, axis=0, keepdims=True)
        w_grad = np.matmul(prev_act.T, delta)

        bias_gradients.append(b_grad)
        weight_gradients.append(w_grad)

        # propagate error to before current layer
        # (omit first layer because it won't be used)
        if i != 0:
            delta = np.matmul(delta, stack.layers[i].weights.T)

    return list(reversed(bias_gradients)), list(reversed(weight_gradients))

class SGD(Optimizer):
    def __init__(self, lr_rate=0.01, momentum=0.9):
        self.lr_rate = lr_rate
        self.momentum = momentum
        self.velocity = None

    def _update_velocity(self, b_grad, w_grad):
            if self.velocity == None:
                self.velocity = {
                    'biases': b_grad,
                    'weights': w_grad
                }
                return

            b_vel = self.velocity['biases']
            w_vel = self.velocity['weights']

            for i in range(len(b_vel)):
                b_vel[i] = b_vel[i] * self.momentum + b_grad[i] * (1-self.momentum)
                w_vel[i] = w_vel[i] * self.momentum + w_grad[i] * (1-self.momentum)

    def apply(self, stack, loss, batch_x, batch_y):
        bias_gradients, weight_gradients = _backprop_gradient(stack, loss, batch_x, batch_y)
        self._update_velocity(bias_gradients, weight_gradients)

        for i, layer in enumerate(stack.layers):
            layer.biases = layer.biases - self.lr_rate * self.velocity['biases'][i]
            layer.weights = layer.weights - self.lr_rate * self.velocity['weights'][i]

        pred = stack.apply(batch_x)
        loss_val = loss(pred, batch_y)

        return stack, loss_val

class Adam(Optimizer):
    def __init__(self, lr_rate=0.01, momentum=0.9):
        self.lr_rate = lr_rate
        self.momentum = momentum
        self.velocity = None
        self.second_velocity = None

    def _update_velocity(self, b_grad, w_grad):
            if self.velocity == None:
                self.velocity = {
                    'biases': b_grad,
                    'weights': w_grad
                }
                self.second_velocity = {
                    'biases': [np.square(layer) for layer in b_grad],
                    'weights': [np.square(layer) for layer in w_grad]
                }
                return

            b_vel = self.velocity['biases']
            w_vel = self.velocity['weights']

            b_vel_2 = self.second_velocity['biases']
            w_vel_2 = self.second_velocity['weights']

            for i in range(len(b_vel)):
                b_vel[i] = b_vel[i] * self.momentum + b_grad[i] * (1-self.momentum)
                w_vel[i] = w_vel[i] * self.momentum + w_grad[i] * (1-self.momentum)

                b_vel_2[i] = b_vel_2[i] * self.momentum + np.square(b_grad[i]) * (1-self.momentum)
                w_vel_2[i] = w_vel_2[i] * self.momentum + np.square(w_grad[i]) * (1-self.momentum)

    def apply(self, stack, loss, batch_x, batch_y):
        bias_gradients, weight_gradients = _backprop_gradient(stack, loss, batch_x, batch_y)
        self._update_velocity(bias_gradients, weight_gradients)

        epsilon = np.finfo(np.float32).eps

        for i, layer in enumerate(stack.layers):

            unbiased_momentum_1 = {
                'biases': self.velocity['biases'][i] / (1 - self.momentum),
                'weights': self.velocity['weights'][i] / (1 - self.momentum)
            }

            unbiased_momentum_2 = {
                'biases': self.second_velocity['biases'][i] / (1 - self.momentum),
                'weights': self.second_velocity['weights'][i] / (1 - self.momentum)
            }

            layer.biases = layer.biases - self.lr_rate * (unbiased_momentum_1['biases']) / (np.sqrt(unbiased_momentum_2['biases']) + epsilon)
            layer.weights = layer.weights - self.lr_rate * (unbiased_momentum_1['weights']) / (np.sqrt(unbiased_momentum_2['weights']) + epsilon)

        pred = stack.apply(batch_x)
        loss_val = loss(pred, batch_y)

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
