import numpy as np
import NNet

def test():
    with open("train-images-idx3-ubyte", "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((num_images, num_rows * num_cols)).transpose()
        images = images.astype(np.float64) / 255
        f.close()
    input_size = 64
    hidden_size = 36
    data = np.random.randn(input_size, 10)
    hidden_size = 96
    L = []
    L.append({"type":'NLayer',"params":{"size": 28 * 28, "ac_func":'sigmoid', "lam": 3e-3,"sparsity_param":0} })
    L.append({"type":'NLayer',"params":{"size": 196, "ac_func":'sigmoid', "lam": 3e-3,"sparsity_param":0.1} })
    L.append({"type":'NLayer',"params":{"size": 28 * 28, "ac_func":'sigmoid', "lam": 3e-3,"sparsity_param":0} })
    ae = NNet(Layers = L,lam = 3e-3, beta = 3)
    ae.train(data = images[:, 0:10000], output = images[:, 0:10000], debug = False)


test()