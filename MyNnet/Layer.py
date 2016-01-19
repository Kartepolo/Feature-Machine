import numpy as np
from scipy.sparse import csr_matrix

class NNode:
    def __init__(self, ac_func):
        self.nodelist = []
        self.func = ac_func

    def add(self, NNode):
        self.nodelist.append(NNode)

    def activate(self):
        #Construct weights and bias arrays
        l = len(self.nodelist)

def softmax(x):
    x = x - np.max(x)
    prob = np.exp(x) / np.sum(np.exp(x), axis = 1)
    return prob


class NLayer:
    def __init__(self, size, ac_func, lam, sparsity_param, prevlayer = None, softmax = False):
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
        self.softmax = softmax

    def rand_init(self, bound = 1):
        r = np.sqrt(6) / np.sqrt(self.size + self.prev.size + 1) * bound
        self.weights = np.random.random((self.size, self.prev.size)) * 2 * r - r
        self.bias = np.zeros(self.size)
        return np.concatenate((self.weights.reshape(self.size * self.prev.size), self.bias))


    def cul_weight(self):
        return self.culmulative_weights

    def activate(self):
        if self.prev is not None:
            prev_activation = self.prev.activate()
            self.sample_size = prev_activation.shape[1]
            if self.softmax:
                z = self.weights.dot(prev_activation)
                self.activation = softmax(z)
            else:
                z = self.weights.dot(prev_activation) + np.tile(self.bias, (self.sample_size, 1)).transpose()
                self.activation, self.deriative = self.func(z)
                self.culmulative_weights = self.prev.cul_weight() + np.sum(self.weights ** 2)
        return self.activation

    def compute_gradient(self, delta):
        w_gradient = delta.dot(self.prev.activation.transpose()) /  self.sample_size + self.lam * self.weights
        b_gradient = np.sum(delta, axis = 1).transpose() / self.sample_size
        return np.concatenate((w_gradient.reshape(self.size * self.prev.size), b_gradient))




