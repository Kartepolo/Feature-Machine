from scipy.stats import norm
from math import sqrt

class AcquisitionFuncs(object):
    '''Returns the acquisition functions.'''
    def __init__(self, k = 1):
        self.kappa = k

    #Methods for calculating single sample

    def EI(self, x, gp, fmax, delta = 1e-3):
        mean,sigma = gp.predict(x, eval_MSE = True)
        if sigma == 0:
            return 0
        else:
            Z = (mean - fmax - delta) / sqrt(sigma)
            return (mean - fmax) * norm.cdf(Z) + sqrt(sigma) * norm.pdf(Z)

    def PoI(self, x, gp, fmax, delta = 1e-3):
        mean,sigma = gp.predict(x, eval_MSE = True)
        if sigma == 0:
            return 1
        else:
            Z = (mean - fmax - delta) / sqrt(sigma)
            return norm.cdf(Z)

    def UCB(self, x, gp, fmax):
        mean,sigma = gp.predict(x, eval_MSE = True)
        return mean + self.kappa * sqrt(sigma)
