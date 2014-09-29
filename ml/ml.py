import numpy as npy
import scipy.io as sio
import scipy.optimize as sop


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


class Learner(object):

    def load_file(self, filename):
        self.training_set = self.training_set_type.from_file(filename)


class SupervisedLearner(Learner):

    def __init__(self, alpha, max_iters, feature_normalization=False):
        self.optimizer = GradDescent(alpha, max_iters)
        self.feature_normalization = feature_normalization
        self.cost_type = Cost
        self.training_set_type = SupervisedData

    def load(self, training_set):
        self.training_set = training_set
        self.cost = self.cost_type(training_set.feature_matrix,
                                   training_set.labels)


class Regression(SupervisedLearner):
    pass


class Classification(SupervisedLearner):
    pass


class LinearRegression(Regression):

    def __init__(self, alpha, max_iters, feature_normalization=False):
        self.optimizer = GradDescent(alpha, max_iters)
        self.feature_normalization = feature_normalization
        self.cost_type = MeanSquaredError

    def learn(self, theta_0=None):
        if theta_0==None:
            n = self.training_set.n
            theta_0 = npy.mat(npy.zeros((n + 1, 1)))
        self.theta, J_history = self.optimizer.optimize(self.cost, theta_0)
        return self.theta, J_history

    def predict(self, data_point):
        training_set = self.training_set
        if training_set.features_are_normalized:
            mu, sigma = training_set.mu, training_set.sigma
            data_point = training_set._normalize_features(data_point,
                                                          mu, sigma)[0]
        augmented_data_point = npy.hstack([npy.mat("1"), data_point])
        return augmented_data_point * self.theta


class LogisticRegression(Classification):
    def __init__(self, max_iters, feature_normalization=False):
        self.cost_type = LogisticCost
        self.feature_normalization = feature_normalization

    def learn(self, theta_0=None):
        raise NotImplementedError()
        if theta_0==None:
            theta_0 = npy.mat(npy.zeros((self.n + 1, 1)))
        self.theta, J_history = sop.optimize(self.cost, theta_0)
        return self.theta, J_history

    def predict(self, data_point):
        raise NotImplementedError()
        

class NeuralNetwork(SupervisedLearner):

    def __init__(self, alpha, max_iters, feature_normalization=False):
        self.optimizer = GradDescent(alpha, max_iters)
        self.feature_normalization = feature_normalization
        self.cost_type = NeuralNetworkCost

    def learn(self, theta_0=None):
        raise NotImplementedError()

    def predict(self, data_point):
        raise NotImplementedError()


def main():
    pass
    
if __name__ == "__main__":
    main()

