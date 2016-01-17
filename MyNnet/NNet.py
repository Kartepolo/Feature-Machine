from Layer import NLayer
import numpy as np
from scipy.optimize import minimize


def kl_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

class NNet:
    def __init__(self, Layers, func, lam, sparsity_param, beta):
        self.lam = lam
        self.sparsity_param = sparsity_param
        self.Layers = []
        self.theta = None
        prev = None
        for l in xrange(len(Layers)):
            self.Layers.append(NLayer(size = Layers[l], ac_func = func, prevlayer = prev, lam = lam, sparsity_param = sparsity_param))
            prev = Layers[l]
        self.beta = beta

    def init(self):
        for l in self.Layers[1:]:
            l.rand_init()

    def feed_back(self, theta, data, debug):
        self.Layers[0].activation = data
        self.set_theta(theta)
        LastLayer = self.Layers[len(self.Layers)]
        result = LastLayer.activate()
        # Sparsity
        rho_est = np.sum(LastLayer.prev.activation, axis = 0) / data.shape[0]
        rho = np.tile(self.sparsity_param, LastLayer.size())
        # Cost Function
        cost = np.sum((result - data) ** 2) / (2 * LastLayer.size()) + \
               self.lam * LastLayer.cul_weight() / len(self.Layers) + \
               self.beta * np.sum(kl_divergence(rho_est, rho))
        sparsity_delta = np.tile(- rho / rho_est + (1 - rho) / (1 - rho_est), (data.shape[0], 1))
        if debug:
            return cost
        grad = []
        delta = -(self.activations[0] - self.activations[2]) * Last
        # Back_Propagation
        for l in xrange(len(self.Layers), 0):
            grad.insert(0, self.Layers[l].compute_gradient())
        return grad

    def set_theta(self, theta):
        cur = 0
        for l in self.Layers[1:]:
            cols = l.size()
            rows = l.prev.size()
            l.weights = theta[cur : (cur + cols * rows)].reshape(rows, cols)
            cur += cols * rows
            l.bias = theta[cur : (cur + cols)]
            cur += cols

    def train(self, data):
        theta = self.init()
        options = {'maxiter': 500, 'disp': True}
        J = lambda x: self.feed_back(theta = x, data = data)
        result = minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
        opt_theta = result.x
        print opt_theta

    def gradient_check(self, data, theta):
        grad0 = self.feed_back(theta, data = data, debug = False)
        epsilon = np.random.random(len(theta)) * 0.0001
        grad1 = np.zeros(theta.shape)
        for i in theta:
            c1 = self.feed_back(theta[i] + epsilon, data = data, debug = True)
            c2 = self.feed_back(theta[i] - epsilon, data = data, debug = True)
            grad1[i] = (c2 - c1)
        diff = np.linalg.norm(grad1 - grad0) / np.linalg.norm(grad1 + grad0)
        print diff







