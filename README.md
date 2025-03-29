# Manual Machine Learning

Written digit recognition network (+ various demos) trained using a DIY ML framework with API inspired by TensorFlow (only Numpy + matplotlib for visualization)

## Features

Currently, the engine supports only linear architecture with dense layers.

The full list of features:

| Type | Features |
| - | - |
| Layers | Dense, Activation |
| Architectures | linear |
| Optimizers | Random, SGD with momentum, Adam |
| Losses | MSE, MAE, Categorical Crossentropy |
| Metrics | MSE, MAE, Accuracy |
| Activations | Linear, ReLU, LeakyReLU, Sigmoid, Tanh, Sine |
| Dataset ops. | shuffling, even/uneven batching |

## Roadmap

Features that will likely be added in the future: (in order of importance)
* Dropout layer
* Conv2D and Flatten layers
* validation datasets
* non-linear architecture with Concatenate layer
* gradient clipping
* Huber loss
* AdamW optimizer

## Dependencies

The engine itself does not rely on anything else than standard python packages and Numpy.

Additional dependencies are required for demos:
* Pillow
* Matplotlib

## Launch

To launch any of the demos, simply clone the repository and execute them:

```sh
git clone https://github.com/mat-kubiak/manual-mnist.git

cd manual-mnist

# (browse `demos` dir for more demos)
python3 -m demos.sin_function
```

## Demos

### 1. Sine Function Reconstruction

![](https://github.com/mat-kubiak/manual-mnist/releases/download/1.0/sin_function.gif)

Trains a network to recreate a $sin(10x)$ function from a single coordinate $x \in [0, 1]$. Demo features a real-time preview of the process.

Code: [`demos/sin_function.py`](demos/sin_function.py)

### 2. Image reconstruction via SIREN

![](https://github.com/mat-kubiak/manual-mnist/releases/download/1.0/image_from_coordinates.gif)

Trains a network to recreate a chosen gray-scale image of any dimensions from two coordinates x and y: $x,y \in [-1, 1]$. The network uses periodic sine activations and a special type of weight initialization with accordance to the SIREN approach. (See: [website](https://www.vincentsitzmann.com/siren/), [paper](https://arxiv.org/abs/2006.09661))

Code: [`demos/image_from_coordinates.py`](demos/image_from_coordinates.py)

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for more info.
