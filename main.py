import numpy as np
import matplotlib.pyplot as plt
import copy

from src.dense_layer import DenseLayer

def mse(x, y):
    return np.mean((x - y)**2)

class neural_net:
    def __init__(self, units=50):
        self.hidden_layer = DenseLayer(1, units, 'relu')
        self.middle_layer = DenseLayer(units, units, 'relu')
        self.output_layer = DenseLayer(units, 1)

    def apply(self, batch):
        x = batch.reshape(-1, 1)
        x = self.hidden_layer.apply(x)
        x = self.middle_layer.apply(x)
        x = self.output_layer.apply(x)
        return x

    def fit(self, batch_x, batch_y):
        preds1 = self.apply(batch_x)
        loss1 = mse(preds1.flatten(), batch_y)

        tries = 3
        best_hidden = None
        best_middle = None
        best_output = None
        best_loss = loss1
        for i in range(tries):
            new_hidden = self.hidden_layer.copy().add_gaussian(stddev=0.035)
            new_middle = self.middle_layer.copy().add_gaussian(stddev=0.035)
            new_output = self.output_layer.copy().add_gaussian(stddev=0.035)

            x = batch_x.reshape(-1, 1)
            x = new_hidden.apply(x)
            x = new_middle.apply(x)
            x = new_output.apply(x)
            loss2 = mse(x.flatten(), batch_y)
            
            if loss2 < best_loss:
                best_hidden = new_hidden
                best_middle = new_middle
                best_output = new_output

        if best_hidden != None:
            self.hidden_layer = best_hidden
            self.middle_layer = best_middle
            self.output_layer = best_output
            print(f'loss: {best_loss}')

def main():

    # create dataset
    x = np.linspace(0, 1, 100, dtype=np.float32)
    y = np.sin(x * 10)

    model = neural_net(units=15)

    batch_size = 10
    num_batches = len(x) // batch_size

    # fitting
    epochs = 10000
    for e in range(epochs):
        model.fit(x, y)

    # evaluating
    preds = np.empty((0,), dtype=np.float32)
    for i in range(num_batches):
        x_batch = x[i*batch_size : (i+1)*batch_size]
        y_batch = y[i*batch_size : (i+1)*batch_size]

        # Apply the neural network on this batch
        x_batch = x_batch.reshape(-1,1)
        predictions = model.apply(x_batch)
        predictions = predictions.flatten()
        preds = np.concatenate([preds, predictions])

        loss = mse(predictions, y_batch)
        print(f'loss: {loss}')

    # visualize results
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    ax.plot(x, preds, linewidth=2.0)
    plt.show()



if __name__ == '__main__':
    main()
