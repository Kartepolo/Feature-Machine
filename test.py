import numpy as np
from Sparse_AutoEncoder import SparseAutoEncoder


def sigmoid(x):
    def func(x):
        return 1.0/(1 + np.exp(-x))
    return func(x), func(x) * (1 - func(x))



def test():

    with open("train-images-idx3-ubyte", "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((num_rows * num_cols, num_images))
        images = images.astype(np.float64) / 255
        f.close()
    input_size = 64
    hidden_size = 36
    data = np.random.randn(input_size, 10)
    SA = SparseAutoEncoder(ac_func = sigmoid, lam = 3e-3, beta = 3, sparsity_param= 0.1)
    SA.train(images[:, 0:100], hidden_size = 36, debug = True)

test()