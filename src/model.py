import numpy as np

from src.layer_stack import LayerStack
from src.losses import get_loss
from src.progress_bar import ProgressBar

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

def great_optimizer(stack, loss, batch_x, batch_y):
    bias_gradients, weight_gradients = real_backprop(stack, batch_x, batch_y)

    lr_rate = 0.0001

    for i, layer in enumerate(stack.layers):
        layer.biases = layer.biases - lr_rate * bias_gradients[i]
        layer.weights = layer.weights - lr_rate * weight_gradients[i]
    
    pred = stack.apply(batch_x)
    loss_val = loss(batch_y, pred)

    return stack, loss_val

def loss_derivative(y_pred, y_true):
    return 2.0*(y_pred-y_true)

def activation_derivative(x):
    return np.where(x > 0, 1.0, 0.0).astype(np.float32)

def real_backprop(stack: LayerStack, x, y_true):
    activations = stack.get_activations(x)
    y_pred = activations[len(activations)-1].flatten()

    bias_gradients = []
    weight_gradients = []

    # initialize delta
    delta = loss_derivative(y_pred, y_true).reshape(-1, 1)

    for i in range(len(stack.layers)-1, -1, -1):
        prev_act = activations[i-1] if i != 0 else x.reshape(-1,1)
        
        current_layer = stack.layers[i]
        prev_layer = stack.layers[i-1]
        weights = current_layer.weights

        if not current_layer.is_activation_linear:
            act_deriv = activation_derivative(prev_act)
            b_grad = np.sum(delta * act_deriv, axis=0, keepdims=False)
        else:
            b_grad = np.sum(delta, axis=0, keepdims=False)

        w_grad = np.matmul(prev_act.T, delta)

        bias_gradients.append(b_grad)
        weight_gradients.append(w_grad)

        if i != 0:
            delta = np.matmul(delta, weights.T)

    return list(reversed(bias_gradients)), list(reversed(weight_gradients))

    # last layer
    b_grad = np.sum(delta, axis=0, keepdims=False)
    w_grad = np.matmul(activations[lnum-2].T, delta)

    bias_gradients.append(b_grad)
    weight_gradients.append(w_grad)

    delta = np.matmul(delta, stack.layers[-1].weights.T)
    
    # second-to-last layer
    act_deriv = activation_derivative(activations[lnum-2])
    
    b_grad = np.sum(delta * act_deriv, axis=0, keepdims=False)
    w_grad = np.matmul(activations[lnum-3].T, delta)

    bias_gradients.append(b_grad)
    weight_gradients.append(w_grad)

    delta = np.matmul(delta, stack.layers[-2].weights.T)
    
    # first layer
    act_deriv = activation_derivative(activations[lnum-3])
    
    b_grad = np.sum(delta * act_deriv, axis=0, keepdims=False)
    w_grad = np.matmul(x.T, delta)

    bias_gradients.append(b_grad)
    weight_gradients.append(w_grad)

    return list(reversed(bias_gradients)), list(reversed(weight_gradients))

class Model:
    def __init__(self, layers, loss):
        self.stack = LayerStack(layers)
        self.loss = get_loss(loss)
        self.optimizer_fn = great_optimizer

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

                self.stack, loss = self.optimizer_fn(self.stack.copy(), self.loss, x_batch, y_batch)
                bar.update_loss(loss)
                losses.append(loss)
                
                if i % 10000 == 0:
                    ebar.update_loss(np.mean(losses))
