import numpy as np

class NNode:
    def __init__(self, ac_func):
        self.nodelist = []
        self.func = ac_func

    def add(self, NNode):
        self.nodelist.append(NNode)

    def activate(self):
        #Construct weights and bias arrays
        l = len(self.nodelist)

def sigmoid(x):
    def func(x):
        return 1.0/(1 + np.exp(-x))
    return func(x), func(x) * (1 - func(x))

class NLayer:
    def __init__(self, size, ac_func, lam, sparsity_param, prevlayer = None):
        self.size = size
        self.func = ac_func
        if prevlayer is not None:
            self.weights = np.zeros(prevlayer.size(), size)
            self.bias = np.zeros(size)
        self.prev = prevlayer
        self.activations = None
        self.initialized = False
        self.sample_size = 0
        self.lam = lam
        self.sparsity_param = sparsity_param
        self.culmulative_weights = 0
        self.deriative = 0

    def rand_init(self, bound = 1):
        r = np.sqrt(6) / np.sqrt(self.size + self.prev.size() + 1) * bound
        self.weights = np.random.random((self.prev.size(), self.size)) * 2 * r - r

    def size(self):
        return self.size

    def cul_weight(self):
        return self.culmulative_weights

    def activate(self):
        if self.prev is not None:
            prev_activations = self.prev.activate()
            self.sample_size = prev_activations.shape[1]
            z = prev_activations.dot(self.weights) + np.tile(self.bias, (self.sample_size, 1))
            self.activations = self.func(z)
            self.culmulative_weights = self.prev.cul_weight + np.sum(self.weights ** 2)
        return self.activations

    def compute_gradient(self, delta):
        w_gradient = self.activations.transpose().dot(delta) /  self.sample_size + self.lam * self.weights[1]
        b_gradient = np.sum(delta, axis = 0).transpose() / self.sample_size
        return np.concatenate((w_gradient.reshape(self.size * self.prev.size()), b_gradient))


