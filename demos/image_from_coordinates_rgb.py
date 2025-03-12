import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.dense_layer import DenseLayer
from src.model import Model
from src import optimizers
from src.initializers import Siren
from src import activations as act

TRAIN_IMAGE = 'images/goldhill.bmp'

def load_image(path):
    image = Image.open(path)
    image = np.array(image).astype(np.float32) / 255.0
    return image

def save_image(img, path):
    # remove alpha
    img = (img[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    pass

def main():

    # create dataset
    img = load_image(TRAIN_IMAGE)
    height, width, _ = img.shape
    y = img.reshape([height * width, 3])

    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)

    X, Y = np.meshgrid(x_coords, y_coords)

    coord_array = np.stack((X, Y), axis=-1)
    x = np.reshape(coord_array, [height*width, 2])

    units = 128
    omega_0 = 30.0

    model = Model(
        loss='mse',
        optimizer=optimizers.Adam(lr_rate=1e-6),
        layers=[
            DenseLayer(2, units, act.Sine(freq=omega_0), initializer=Siren(omega_0=omega_0, is_first=True)),
            DenseLayer(units, units, act.Sine(freq=omega_0), initializer=Siren(omega_0=omega_0)),
            DenseLayer(units, units, act.Sine(freq=omega_0), initializer=Siren(omega_0=omega_0)),
            DenseLayer(units, units, act.Sine(freq=omega_0), initializer=Siren(omega_0=omega_0)),
            DenseLayer(units, units, act.Sine(freq=omega_0), initializer=Siren(omega_0=omega_0)),
            DenseLayer(units, 3, 'sigmoid', initializer=Siren())
        ]
    )

    def save_progress_image(epoch):
        preds = model.apply(x).reshape([height, width, 3])
        save_image(preds, f'animation/{epoch}.png')

    stats = model.fit(x, y,
        batch_size=64,
        epochs=500,
        epoch_callback=save_progress_image
    )

    preds = model.apply(x).reshape([height, width, 3])
    loss = model.loss(preds.flatten(), img.flatten())
    print(f"Final loss ({model.loss.get_name()}): {loss:.4e}")
    print(f"Mean epoch duration: {stats['mean_epoch_duration']:.2f} s")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    axes[0].imshow(img, vmin=0, vmax=1)
    axes[0].set_title("True")
    axes[0].axis("off")

    axes[1].imshow(preds, vmin=0, vmax=1)
    axes[1].set_title("Predicted")
    axes[1].axis("off")

    loss_history = stats['loss_history']
    axes[2].plot(range(len(loss_history)), loss_history, color='red', linewidth=2.0)
    axes[2].set_title("Loss History")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_yscale('log')
    axes[2].grid()

    plt.show()

if __name__ == '__main__':
    main()
