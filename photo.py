import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    colored = cm.get_cmap(CMAP)(img)

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

    units = 50
    model = Model(
        loss='mse',
        optimizer=optimizers.SGD(lr_rate=0.01),
        layers=[
            DenseLayer(2, units, 'leaky_relu'),
            DenseLayer(units, units, 'leaky_relu'),
            DenseLayer(units, units, 'leaky_relu'),
            DenseLayer(units, units, 'leaky_relu'),
            DenseLayer(units, units, 'tanh'),
            DenseLayer(units, 1, 'sigmoid')
        ]
    )

    batch_size = 15
    num_batches = len(x) // batch_size

    for i in range(200):
        loss_history = model.fit(x, y, batch_size=batch_size, epochs=1)

        preds = model.apply(x).reshape([height, width])
        save_image(preds, f'animation/{i}.png')

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
