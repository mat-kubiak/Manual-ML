import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.dense_layer import DenseLayer
from src.model import Model
from src import optimizers

CMAP = 'inferno'
TRAIN_IMAGE = 'images/lenna.jpg'

def load_image(path):
    image = Image.open(path).convert("L")  # Convert to grayscale
    image = np.array(image).astype(np.float32) / 255.0
    return image

def save_image(img, path):
    colored = plt.get_cmap(CMAP)(img)

    # remove alpha
    colored = (colored[:, :, :3] * 255).astype(np.uint8)

    Image.fromarray(colored).save(path)

def main():

    # create dataset
    img = load_image(TRAIN_IMAGE)
    height, width = img.shape
    y = img.flatten()

    x_coords = np.linspace(0.0, 1.0, width)
    y_coords = np.linspace(0.0, 1.0, height)

    X, Y = np.meshgrid(x_coords, y_coords)

    coord_array = np.stack((X, Y), axis=-1)
    x = np.reshape(coord_array, [height*width, 2])

    units = 100
    model = Model(
        loss='mse',
        optimizer=optimizers.Adam(lr_rate=1e-4),
        layers=[
            DenseLayer(2, units, 'leaky_relu'),
            DenseLayer(units, units, 'leaky_relu'),
            DenseLayer(units, units, 'leaky_relu'),
            DenseLayer(units, units, 'leaky_relu'),
            DenseLayer(units, units, 'tanh'),
            DenseLayer(units, 1, 'sigmoid')
        ]
    )

    def save_progress_image(epoch):
        preds = model.apply(x).reshape([height, width])
        save_image(preds, f'animation/{epoch}.png')

    loss_history = model.fit(x, y,
        batch_size=32,
        epochs=1000,
        epoch_callback=save_progress_image
    )

    preds = model.apply(x).reshape([height, width])
    loss = model.loss(preds.flatten(), img.flatten())
    print(f"Final loss ({model.loss.get_name()}): {loss}")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    axes[0].imshow(img, cmap=CMAP, vmin=0, vmax=1)
    axes[0].set_title("True")
    axes[0].axis("off")

    axes[1].imshow(preds, cmap=CMAP, vmin=0, vmax=1)
    axes[1].set_title("Predicted")
    axes[1].axis("off")

    plt.show()

if __name__ == '__main__':
    main()
