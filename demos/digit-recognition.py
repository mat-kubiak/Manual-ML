import os, hashlib, urllib.request
from pathlib import Path
import numpy as np

class DLProgbar:
    def __init__(self):
        self.progbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.progbar:
            self.progbar = DownloadProgressBar(total_iters=total_size)

        current = block_num * block_size
        if current < total_size:
            self.progbar.update(current)
        else:
            self.progbar.close()

def download_mnist():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    true_hash = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
    dpath = './datasets/mnist.npz'

    if Path(dpath).is_file():
        print('MNIST dataset already downloaded, skipping...')
    else:
        print('Downloading MNIST dataset...')

        Path("./datasets").mkdir(parents=True, exist_ok=True)
        dpath, _ = urllib.request.urlretrieve(url=url, filename=dpath, reporthook=DLProgbar())
        print('Download successful!')

        pred_hash = hashlib.sha256()
        with open(dpath,"rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096),b""):
                pred_hash.update(byte_block)
        if pred_hash.hexdigest() != true_hash:
            print('Hash check did not succeed, please try again!')
            print(f'correct:  {true_hash}')
            print(f'computed: {pred_hash.hexdigest()}')
            os.remove(dpath)
            exit()

    with np.load(dpath, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)

def main():

    (x_train, y_train), (x_test, y_test) = download_mnist()

    # flatten images
    x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))

    # normalize to [0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # one-hot encode
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

if __name__ == '__main__':
    main()
