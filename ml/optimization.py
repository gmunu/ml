import numpy as npy

from cost import *

class Optimizer(object):
    pass

class GradDescent(Optimizer):

    def __init__(self, alpha, max_iters, epsilon=None):
        self.alpha = alpha
        self.max_iters = max_iters
        self.epsilon = epsilon

    def optimize(self, cost, theta_0):
        theta = theta_0.copy()
        J_history = npy.zeros((self.max_iters + 1, 1))
        J_history[0] = cost.compute(theta)
        for i in xrange(self.max_iters):
            grad = cost.grad(theta)
            theta = theta - self.alpha * grad
            J_history[i + 1] = cost.compute(theta)
        return theta, J_history



