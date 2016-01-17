import numpy as np
from scipy.optimize import minimize


def kl_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


class SparseAutoEncoder:


    def __init__(self, ac_func, lam, sparsity_param, beta):
        self.layout = (0, 0)
        self.weights = dict()
        self.bias = dict()
        self.activations = dict()
        self.deriatives = dict()
        self.func = ac_func
        self.sparsity_param = sparsity_param
        self.lam = lam
        self.beta = beta
        self.sample_size = 0

    def initialize(self, train, hidden_size):
        self.layout = (train.shape[1], hidden_size)
        r = np.sqrt(6) / np.sqrt(sum(self.layout) + 1)
        self.weights[0] = np.random.random((self.layout[0], self.layout[1])) * 2 * r - r
        self.weights[1] = np.random.random((self.layout[1], self.layout[0])) * 2 * r - r
        self.bias[0] = np.zeros(self.layout[1])
        self.bias[1] = np.zeros(self.layout[0])
        self.sample_size = train.shape[0]
        self.activations[0] = train
        l = self.layout[0] * self.layout[1]
        return np.concatenate((self.weights[0].reshape(l), self.weights[1].reshape(l), self.bias[0], self.bias[1]))

    def set_theta(self, theta):
        l = self.layout[0] * self.layout[1]
        self.weights[0] = theta[0:l].reshape(self.layout[0], self.layout[1])
        self.weights[1] = theta[l:2 * l].reshape(self.layout[1], self.layout[0])
        self.bias[0] = theta[2 * l: 2 * l + self.layout[1]]
        self.bias[1] = theta[2 * l + self.layout[1] :]

    def forward_back(self, theta, debug = False):
        self.set_theta(theta)
        for i in self.weights.keys():
            z = self.activations[i].dot(self.weights[i]) + np.tile(self.bias[i], (self.sample_size, 1))
            self.activations[i + 1], self.deriatives[i + 1] = self.func(z)
        # Sparsity
        rho_est = np.sum(self.activations[1], axis = 0) / self.sample_size
        rho = np.tile(self.sparsity_param, self.layout[1])
        # Cost Function
        cost = np.sum((self.activations[2] - self.activations[0]) ** 2) / (2 * self.sample_size) + \
               self.lam * (np.sum(self.weights[0] ** 2) + np.sum(self.weights[1] ** 2)) / 2 + \
               self.beta * np.sum(kl_divergence(rho, rho_est))
        sparsity_delta = np.tile(- rho / rho_est + (1 - rho) / (1 - rho_est), (self.sample_size, 1))
        if debug:
            return cost
        w_gradient = {}
        b_gradient = {}
        # Back_Propagation
        delta3 = -(self.activations[0] - self.activations[2]) * self.deriatives[2]
        delta2 = (delta3.dot(self.weights[1].transpose()) + self.beta * sparsity_delta) * self.deriatives[1]
        w_gradient[1] = self.activations[1].transpose().dot(delta3) / self.sample_size + self.lam * self.weights[1]
        w_gradient[0] = self.activations[0].transpose().dot(delta2) / self.sample_size + self.lam * self.weights[0]
        b_gradient[1] = np.sum(delta3, axis = 0).transpose() / self.sample_size
        b_gradient[0] = np.sum(delta2, axis = 0).transpose() / self.sample_size
        l = self.layout[0] * self.layout[1]
        grad = np.concatenate((w_gradient[0].reshape(l), w_gradient[1].reshape(l), b_gradient[0], b_gradient[1]))
        print cost, grad
        return cost, grad

    def train(self, train, hidden_size, debug = True):
        theta = self.initialize(train, hidden_size)
        if debug:
            self.check_gradient(theta)
        else:
            options = {'maxiter': 500, 'disp': True}
            J = lambda x: self.forward_back(theta = x, debug = False)
            result = minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
            opt_theta = result.x
            print opt_theta

    def check_gradient(self, theta):
        grad0 = self.forward_back(theta, debug = False)[1]
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
            print "Computing gradient for input:", i
        print "Printing difference \n"
        diff = np.linalg.norm(grad1 - grad0) / np.linalg.norm(grad1 + grad0)
        print diff
        print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n"


















