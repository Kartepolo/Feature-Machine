import Layer as NLayer
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix

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

def sign(x,y):
    return np.int64((x - y) == 0)

class NNet(object):
    def __init__(self, Layers, beta):
        self.Layers = [] # Layers is a list of dictionary
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


    def predict(self, data):
        self.Layers[0].activation = data
        LastLayer = self.Layers[len(self.Layers) - 1]
        result = LastLayer.activate()
        return LastLayer.pred

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
        cost = LastLayer.cost(self.output, s_cost, len(self.Layers) - 1)
        if debug:
            return cost
        grad = []
        if LastLayer.funcname == 'softmax':
            indicator = csr_matrix((np.ones(LastLayer.sample_size), (self.output, np.array(range(LastLayer.sample_size)))))
            indicator = np.array(indicator.todense())
            delta = -(indicator - LastLayer.activation)
        else:
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


    def set_input(self, data):
        self.Layers[0].activation = data

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
            options = {'maxiter': 200, 'disp': True}
            J = lambda x: self.forward_back(theta = x, debug = debug)
            result = minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
            opt_theta = result.x
        return True

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

    def naive_SGD(self, data, output, l_rate,decay = 1, batches = 1, Uafterbatch = True,debug = True):
        acc_cost = {'cost':[], 'variance':[]}
        theta = self.init()
        err = float('Inf')
        counter = 0
        batch_size = (int)(1.0 * data.shape[1] / batches)
        batch_num = 0
        momentum = [0]
        gamma = 0.5
        drop = False
        while (err > 1e-1 and batch_num < 100):
            for i in xrange(batches):
                if i == batches - 1 : upper = data.shape[1]
                else: upper = (i+1) * batch_size
                self.Layers[0].activation = data[:,i * batch_size : upper]
                self.output = output[i * batch_size : upper].T
                cost, grad = self.forward_back(theta, debug = False)
                momentum.append(gamma * momentum[counter] + l_rate * grad)
                counter = counter + 1
                theta = theta - momentum[counter]
                if not Uafterbatch:
                    l_rate = l_rate * decay
            if Uafterbatch:
                l_rate = l_rate * decay
                gamma = min(gamma * 1.01 + 0.02, gamma * 1.05)
            acc_cost['cost'].append(1.0 * sum(self.J) / len(self.J))
            self.J = []
            batch_num += 1
            #print "Batch", batch_num, "Complete, using learning rate: ", l_rate
            #print "Mean: ",mean, "max_val: ", max_val, "min_val: ", min_val
            delta = []
            if len(acc_cost) > 1:
                for i in xrange(len(acc_cost['cost'])-1):
                    delta.append(acc_cost['cost'][i] - acc_cost['cost'][i + 1])
            #print "Delta" , len(delta) == 0 or delta[len(delta)-1]
            if (len(delta) > 0):
                if (delta[len(delta) -1] < 1e-6):
                    break
        return acc_cost


class Sparse_AutoEncoder(NNet):


    def __init__(self, input_size, hidden_size, lam, sparsity_param, beta):
        L = [{'type':'NLayer', 'params' : {"size": input_size, "ac_func":'sigmoid', "lam": lam,"sparsity_param":0}},
             {'type':'NLayer', 'params' : {"size": hidden_size, "ac_func":'sigmoid', "lam": lam,"sparsity_param":sparsity_param}},
             {'type':'NLayer', 'params' : {"size": input_size, "ac_func":'sigmoid', "lam": lam,"sparsity_param":0}}]
        super(Sparse_AutoEncoder, self).__init__(L,beta)

    def train(self, data):
        super(Sparse_AutoEncoder, self).train(data, data,debug = False)








