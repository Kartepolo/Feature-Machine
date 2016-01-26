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

class NLayer(object):
    def __init__(self, size, ac_func, lam, sparsity_param, prevlayer = None):
        self.size = size
        self.funcname = ac_func
        self.type = type
        self.sparsity_delta = None
        self.prev = prevlayer
        self.activation = None
        self.sparsity_param = sparsity_param
        if self.prev is not None:
            self.weights = np.empty((self.size, self.prev.size))
        self.sample_size = 0
        self.lam = lam
        self.culmulative_weights = 0
        self.derivative = 0
        self.rho = np.tile(self.sparsity_param, self.size)
        self.rho_est = None

    def sigmoid(self,x):
        def func(x):
            return 1.0/(1 + np.exp(-1.0 * x))
        return func(x), func(x) * (1 - func(x))


    def rand_init(self):
        r = np.sqrt(6) / np.sqrt(self.size + self.prev.size + 1)
        self.weights = np.random.random((self.size, self.prev.size)) * 2 * r - r
        self.bias = np.zeros(self.size,dtype=np.float64)
        return np.concatenate((self.weights.reshape(self.size * self.prev.size), self.bias))


    def activate(self):
        if self.prev is not None:
            prev_activation = self.prev.activate()
            self.sample_size = prev_activation.shape[1]
            z = self.weights.dot(prev_activation) + np.tile(self.bias, (self.sample_size, 1)).transpose()
            func = getattr(self,self.funcname)
            self.activation, self.derivative = func(z)
            self.culmulative_weights = self.prev.culmulative_weights + np.sum(self.weights ** 2)
            self.rho_est = np.sum(self.activation, axis = 1) / self.sample_size
            self.sparsity_delta = np.tile(- self.rho / self.rho_est + (1 - self.rho) / (1 - self.rho_est), (self.sample_size, 1)).transpose()
        return self.activation

    def compute_gradient(self, delta):
        w_gradient = delta.dot(self.prev.activation.transpose()) / self.sample_size + self.lam * self.weights
        b_gradient = np.sum(delta, axis = 1).transpose() / self.sample_size
        next_delta = 0
        if self.prev is not None:
            next_delta = self.weights.transpose().dot(delta) * self.prev.derivative
        return (np.concatenate((w_gradient.reshape(self.size * self.prev.size), b_gradient)), next_delta)


class Encoder(NLayer):

    def __init__(self, sparsity_param, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.sparsity_param = sparsity_param
        self.bias = np.empty(self.size,dtype=np.float64)
        self.rho = None
        self.rho_est = None


    def compute_gradient(self, delta):
        # Sparsity
        self.rho_est = np.sum(self.prev.activation, axis = 1) / self.sample_size
        self.rho = np.tile(self.sparsity_param, self.prev.size)
        sparsity_delta = np.tile(- self.rho / self.rho_est + (1 - self.rho) / (1 - self.rho_est), (self.sample_size, 1)).transpose()
        w_gradient = delta.dot(self.prev.activation.transpose()) /  self.sample_size + self.lam * self.weights
        b_gradient = np.sum(delta, axis = 1).transpose() / self.sample_size
        next_delta = 0
        if self.prev is not None:
            next_delta = self.weights.transpose().dot(delta) * self.prev.deriative + self.sparsity_param * sparsity_delta
        return (np.concatenate((w_gradient.reshape(self.size * self.prev.size), b_gradient)), next_delta)


class Dropout(NLayer):

    def rand_init(self):
        r = np.random.randint(self.prev.size, size = np.floor(self.prev.size))
        indice = zip(xrange(self.size), r)
        self.weights[indice] = 1

    def activate(self):
        if self.prev is not None:
            prev_activation = self.prev.activate()
            self.sample_size = prev_activation.shape[1]
            self.activation = self.weights.dot(prev_activation)
            self.derivative = 1
            self.culmulative_weights = self.prev.culmulative_weights
        return self.activation

    def compute_gradient(self, delta):
        w_gradient = delta.dot(self.prev.activation.transpose()) /  self.sample_size + self.lam * self.weights
        next_delta = 0
        if self.prev is not None:
            next_delta = self.weights.transpose().dot(delta) * self.prev.deriative
        return (w_gradient.reshape(self.size * self.prev.size), next_delta)

class classifier(NLayer):

    def __init__(self, **kwargs):
        super(classifier, self).__init__(**kwargs)
        self.bias = np.empty(self.size,dtype=np.float64)

    def activate(self):
        prev_activation = self.prev.activate()
        self.sample_size = prev_activation.shape[1]
        z = self.weights.dot(prev_activation)
        func = getattr(self,self.funcname)
        self.activation = func(z)
        self.culmulative_weights = self.prev.culmulative_weights + np.sum(self.weights ** 2)

class Regressor(NLayer):

    def __init__(self, **kwargs):
        super(Regressor, self).__init__(**kwargs)
        assert self.size == 1
        self.bias = np.ones(self.size,dtype=np.float64)

    def rand_init(self):
        self.weights = np.random.random((self.size, self.prev.size)) * 2 * 1 - 1
        self.bias = np.zeros(self.size,dtype=np.float64)
        return np.concatenate((self.weights.reshape(self.size * self.prev.size), self.bias))

    def activate(self):
        if self.prev is not None:
            prev_activation = self.prev.activate()
            self.sample_size = prev_activation.shape[1]
            self.activation = self.weights.dot(prev_activation) + np.tile(self.bias, (self.sample_size, 1)).transpose()
            self.derivative = 1
            self.culmulative_weights = self.prev.culmulative_weights + np.sum(self.weights ** 2)
        return self.activation

    def compute_gradient(self, delta):
        w_gradient = delta.dot(self.prev.activation.transpose()) / self.sample_size + self.lam * self.weights
        b_gradient = np.sum(delta, axis = 1) / self.sample_size
        return (np.append(w_gradient, b_gradient), delta)










