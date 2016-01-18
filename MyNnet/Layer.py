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


class NLayer:
    def __init__(self, size, ac_func, lam, sparsity_param, prevlayer = None):
        self.size = size
        self.func = ac_func
        self.prev = prevlayer
        self.activation = None
        self.initialized = False
        self.sample_size = 0
        self.lam = lam
        self.sparsity_param = sparsity_param
        self.culmulative_weights = 0
        self.deriative = 0

    def rand_init(self, bound = 1):
        r = np.sqrt(6) / np.sqrt(self.size + self.prev.size + 1) * bound
        self.weights = np.random.random((self.prev.size, self.size)) * 2 * r - r
        self.bias = np.zeros(self.size)
        return np.concatenate((self.weights.reshape(self.size * self.prev.size), self.bias))


    def cul_weight(self):
        return self.culmulative_weights

    def activate(self):
        if self.prev is not None:
            prev_activation = self.prev.activate()
            self.sample_size = prev_activation.shape[0]
            z = prev_activation.dot(self.weights) + np.tile(self.bias, (self.sample_size, 1))
            self.activation, self.deriative = self.func(z)
            self.culmulative_weights = self.prev.cul_weight() + np.sum(self.weights ** 2)
        return self.activation

    def compute_gradient(self, delta):
        w_gradient = self.prev.activation.transpose().dot(delta) /  self.sample_size + self.lam * self.weights
        b_gradient = np.sum(delta, axis = 0).transpose() / self.sample_size
        return np.concatenate((w_gradient.reshape(self.size * self.prev.size), b_gradient))


