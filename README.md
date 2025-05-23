# Manual Machine Learning

MNIST written digit recognition (+ various other demos) trained using a DIY ML framework with TensorFlow-inspired API

## Features

Currently, the framework supports only linear architecture with dense layers.

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

The framework itself does not rely on anything else than Numpy, Tqdm and standard python packages.

Additional dependencies are required for launching demos:
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

Trains a network to recreate a $sin(10x)$ function from a single coordinate $x \in [0, 1]$. Demo features a real-time preview of the process.

Code: [`demos/sin_function.py`](demos/sin_function.py)

https://github.com/user-attachments/assets/19e969d7-7b87-4dc7-a0bb-6690ba88af27

### 2. Image reconstruction via SIREN

Trains a network to recreate a chosen gray-scale or rgb image of any dimensions from two coordinates x and y: $x,y \in [-1, 1]$. The network uses periodic sine activations and a special type of weight initialization with accordance to the SIREN approach. (See: [website](https://www.vincentsitzmann.com/siren/), [paper](https://arxiv.org/abs/2006.09661))

Code: [`demos/image_from_coordinates.py`](demos/image_from_coordinates.py) and [`demos/image_from_coordinates_rgb.py`](demos/image_from_coordinates_rgb.py)

__Example gray-scale image (colored with cmap):__ 

https://github.com/user-attachments/assets/8ef61d7d-5f81-4562-bf19-d10c7a8b75fe

__Example rgb image:__

https://github.com/user-attachments/assets/43d97b44-ec67-4dac-832e-0748fb44bb03

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for more info.
