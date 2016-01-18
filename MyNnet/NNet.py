from Layer import NLayer
import numpy as np
from scipy.optimize import minimize

def sigmoid(x):
    def func(x):
        return 1.0/(1 + np.exp(-x))
    return func(x), func(x) * (1 - func(x))

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
            prev = self.Layers[l]
        self.beta = beta

    def init(self):
        result = []
        for l in self.Layers[1:]:
            result.append(l.rand_init())
        return np.hstack(result)


    def forward_back(self, theta, debug = True):
        self.set_theta(theta)
        LastLayer = self.Layers[len(self.Layers) - 1]
        result = LastLayer.activate()
        # Sparsity
        rho_est = np.sum(LastLayer.prev.activation, axis = 0) / LastLayer.sample_size
        rho = np.tile(self.sparsity_param, LastLayer.prev.size)
        # Cost Function
        cost = np.sum((result - self.output) ** 2) / (2 * LastLayer.sample_size) + \
               self.lam * LastLayer.cul_weight() / (len(self.Layers) - 1) + \
               self.beta * np.sum(kl_divergence(rho, rho_est))
        sparsity_delta = np.tile(- rho / rho_est + (1 - rho) / (1 - rho_est), (LastLayer.sample_size, 1))
        if debug:
            return cost
        grad = []
        delta = -(self.output - LastLayer.activation) * LastLayer.deriative
        # Back_Propagation
        for l in xrange(len(self.Layers) - 1, 0, -1):
            grad.insert(0, self.Layers[l].compute_gradient(delta))
            delta = (delta.dot(self.Layers[l].weights.transpose())) * self.Layers[l - 1].deriative
        return np.hstack(grad)

    def set_theta(self, theta):
        cur = 0
        for l in self.Layers[1:]:
            cols = l.size
            rows = l.prev.size
            l.weights = theta[cur : (cur + cols * rows)].reshape(rows, cols)
            cur += cols * rows
            l.bias = theta[cur : (cur + cols)]
            cur += cols

    def train(self, data, output, debug = True):
        self.output = output
        theta = self.init()
        self.Layers[0].activation = data
        if debug:
            self.gradient_check(data, theta)
        else:
            options = {'maxiter': 500, 'disp': True}
            J = lambda x: self.forward_back(theta = x)
            result = minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
            opt_theta = result.x
            print opt_theta

    def gradient_check(self, data, theta):
        grad0 = self.forward_back(theta, debug = False)
        epsilon = 0.0001
        grad1 = np.zeros(theta.shape)
        for i in range(theta.shape[0]):
            theta_epsilon_plus = np.array(theta, dtype=np.float64)
            theta_epsilon_plus[i] = theta[i] + epsilon
            theta_epsilon_minus = np.array(theta, dtype=np.float64)
            theta_epsilon_minus[i] = theta[i] - epsilon
            c1 = self.forward_back(theta_epsilon_plus, debug = True)
            c2 = self.forward_back(theta_epsilon_minus, debug = True)
            grad1[i] = (c1 - c2) / (2 * epsilon)
        diff = np.linalg.norm(grad1 - grad0) / np.linalg.norm(grad1 + grad0)
        print diff

def test():
    input_size = 64
    data = np.random.randn(100, input_size)
    ae = NNet(Layers = [64,36,64], func = sigmoid, lam = 3e-3, beta = 3, sparsity_param= 0.1)
    ae.train(data,data)

test()






