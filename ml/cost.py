import numpy as npy

class Cost:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)

    def compute(self, theta):
        raise NotImplementedError()

    def grad(self, theta):
        raise NotImplementedError()


class MeanSquaredError(Cost):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)

    def compute(self, theta):
        h = self.X * theta # hypothesis
        return (h - self.y).T * (h - self.y) / (2 * self.m)

    def grad(self, theta):
        h = self.X * theta
        return (1.0 / self.m * self.X.T * (h - self.y))


def sigmoid(A):
    return 1.0 / (1 + npy.exp(-A))


class LogisticCost(Cost):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)

    def compute(self, theta, regularization_const=0.0):
        h = sigmoid(self.X * theta)
        J = -1.0 / self.m * (self.y.T * npy.log(h)
                             + (1 - self.y).T * npy.log(1 - h))
        # regularization:
        if regularization_const != 0.0:
            lmbda = float(regularization_const)
            theta_no_bias = theta[1:, :]
            J += lmbda / (2 * self.m) * (theta_no_bias.T * theta_no_bias)
        return J

    def grad(self, theta):
        h = sigmoid(self.X * theta)
        return (1.0 / self.m * self.X.T * (h - self.y))


class NeuralNetworkCost(Cost):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)

    def compute(self, thetas, regularization_const=0.0):
        # thetas is a tuple of theta-matrices
        L = len(thetas) # number of layers
        K = thetas[-1].shape[0] # number of classes
                                #   = number of rows in last theta
        # hypothesis computation via feed propagation:
        a_curr = self.X.T # already includes the bias row
        a_list = [a_curr]
        for theta in thetas:
            z = theta * a_curr
            a_curr = sigmoid(z)
            bias_row = npy.mat(npy.ones((1, self.m)))
            a_curr = npy.vstack([bias_row, a_curr])
            a_list.append(a_curr)
        h = a_list[-1][1:, :]
        # create indicator row in Y, for each exmple label in y:
        Y = npy.zeros((self.m, K))
        Y[npy.hstack(map(lambda i: self.y == i, xrange(1, K + 1)))] = 1
        # cost computation:
        Y_flat = Y.flatten() # a row!
        h_T_flat = h.T.flatten()
        J = Y_flat * (npy.log(h_T_flat).T)
        J += (1 - Y_flat) * (npy.log(1 - h_T_flat).T)
        J *= - 1.0 / self.m
        # regularization:
        if regularization_const != 0.0:
            lmbda = float(regularization_const)
            for theta in thetas:
                # flat_theta = theta.flatten()
                flat_theta = theta[:, 1:].flatten()
                J += lmbda / (2 * self.m) * flat_theta * flat_theta.T
        return J



