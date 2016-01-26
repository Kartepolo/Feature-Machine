import Layer as NLayer
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle


def sigmoid(x):
    def func(x):
        return 1.0/(1 + np.exp(-x))
    return func(x), func(x) * (1 - func(x))

def kl_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def softmax(x):
    x = x - np.max(x)
    prob = np.exp(x) / np.sum(np.exp(x), axis = 1)
    return prob

class NNet:
    def __init__(self, Layers, lam, beta):
        self.lam = lam
        self.Layers = [] # Layers is a list of dictionary
        self.theta = None
        self.beta = beta
        prev = None
        for l in xrange(len(Layers)):
            name = Layers[l]['type']
            params = Layers[l]['params']
            params['prevlayer'] = prev
            method = getattr(NLayer,name)
            self.Layers.append(method(**params))
            prev = self.Layers[l]
        # a list to store costs
        self.J = []

    def init(self):
        result = []
        for l in self.Layers[1:]:
            result.append(l.rand_init())
        return np.hstack(result)

    def forward_back(self, theta, debug = True):
        self.set_theta(theta)
        LastLayer = self.Layers[len(self.Layers) - 1]
        result = LastLayer.activate()
        #Aggegating kl_divergence of sparsity_parameters
        s_cost = 0
        for l in self.Layers:
            if l.sparsity_param > 0:
                s_cost += self.beta* np.sum(kl_divergence(l.rho, l.rho_est))
        # Cost Function
        cost = np.sum((result - self.output) ** 2) / (2 * LastLayer.sample_size) \
               + self.lam * LastLayer.culmulative_weights / (len(self.Layers) - 1) +s_cost
        if debug:
            return cost
        grad = []
        delta = -(self.output - LastLayer.activation) * LastLayer.derivative
        # Back_Propagation
        for l in xrange(len(self.Layers) - 1, 0, -1):
            if self.Layers[l].sparsity_param > 0:
                sparsity_delta = np.tile(- self.Layers[l].rho / self.Layers[l].rho_est
                                         + (1 - self.Layers[l].rho) / (1 - self.Layers[l].rho_est),
                                         (self.Layers[l].sample_size, 1)).transpose()
                delta = delta + self.beta * sparsity_delta * self.Layers[l].derivative
            lgrad, delta = self.Layers[l].compute_gradient(delta)
            grad.insert(0, lgrad)
        self.J.append(cost)
        return cost, np.hstack(grad)

    def set_theta(self, theta):
        cur = 0
        for l in self.Layers[1:]:
            cols = l.size
            rows = l.prev.size
            l.weights = theta[cur : (cur + cols * rows)].reshape(cols, rows)
            cur += cols * rows
            l.bias = theta[cur : (cur + cols)]
            cur += cols

    def train(self, data, output, debug = True):
        #data = normalize(data, axis=0, norm='l1')
        #output = normalize(output, axis=0, norm='l1')
        self.Layers[0].activation = data
        self.output = output
        theta = self.init()
        if debug:
            self.gradient_check(data, theta)
        else:
            options = {'maxiter': 500, 'disp': True}
            J = lambda x: self.forward_back(theta = x, debug = debug)
            result = minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
            opt_theta = result.x
            print opt_theta

    def gradient_check(self, data, theta):
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
        diff = np.linalg.norm(grad1 - grad0) / np.linalg.norm(grad1 + grad0)
        print diff

    def naive_SGD(self, data, output, l_rate,decay = 1, norm_method = 'max', batches = 10, Uafterbatch = True):
        acc_cost = {'cost':[], 'variance':[]}
        if norm_method == 'max':
            data = (data.T / (data.max(axis=1) + np.spacing(0))).T
        else:
            data = normalize(data, axis=0, norm='l1')
        theta = self.init()
        err = float('Inf')
        counter = 0
        batch_size = (int)(1.0 * data.shape[1] / batches)
        batch_num = 0
        while (err > 1e-1 and batch_num < 50):
            for i in xrange(batches):
                if i == batches - 1 : upper = data.shape[1]
                else: upper = (i+1) * batch_size
                self.Layers[0].activation = data[:,i * batch_size : upper]
                self.output = output[i * batch_size : upper]
                cost, grad = self.forward_back(theta, debug = False)
                theta = theta - l_rate * grad
                counter = counter + 1
                if not Uafterbatch:
                    l_rate = l_rate * decay
            if Uafterbatch:
                l_rate = l_rate * decay
            mean = float(sum(self.J))/ (batches + 1)
            max_val = max(self.J)
            min_val = min(self.J)
            self.J = []
            batch_num += 1
            acc_cost['cost'].append(mean)
            acc_cost['variance'].append(max_val - min_val)
            #print "Batch", batch_num, "Complete, using learning rate: ", l_rate
            #print "Mean: ",mean, "max_val: ", max_val, "min_val: ", min_val
            delta = []
            if len(acc_cost) > 1:
                for i in xrange(len(acc_cost['cost'])-1):
                    delta.append(acc_cost['cost'][i] - acc_cost['cost'][i + 1])
            #print "Delta" , len(delta) == 0 or delta[len(delta)-1]
            if (len(delta) > 0):
                if (delta[len(delta) -1] < 1e-3):
                    break
        return acc_cost


class Sparse_AutoEncoder:


    def __init__(self, input_size, hidden_size, lam, sparsity_param, beta):
        self.ae = NNet(Layers = [input_size ,hidden_size, input_size], func = sigmoid, lam = lam, beta = beta, sparsity_param= sparsity_param)

    def train(self, data):
        self.ae.train(data, data, debug = False)








