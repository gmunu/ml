import numpy as np


class Cost:
    def compute(self, theta):
        pass

    def grad(self, theta):
        pass

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


class LogisticCost(Cost):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)

    def _sigmoid(self, A):
        return 1.0 / (1 + np.exp(-A))

    def compute(self, theta):
        h = self._sigmoid(self.X * theta)
        return (-1.0 / self.m * (self.y.T * np.log(h) 
                             + (1 - self.y).T * np.log(1 - h)))

    def grad(self, theta):
        h = self._sigmoid(self.X * theta)
        return (1.0 / self.m * self.X.T * (h - self.y))


class Optimizer(object):
    pass

class GradDescent(Optimizer):
    def __init__(self, alpha, max_iters, epsilon=None):
        self.alpha = alpha
        self.max_iters = max_iters
        self.epsilon = epsilon

    def optimize(self, cost, theta_0):
        theta = theta_0.copy()
        J_history = np.zeros((self.max_iters + 1, 1))
        J_history[0] = cost.compute(theta)
        for i in xrange(self.max_iters):
            grad = cost.grad(theta)
            theta = theta - self.alpha * grad
            J_history[i + 1] = cost.compute(theta)
        return theta, J_history


class SupervisedLearner(object):
    def load(self, filename):
        feature_matrix, labels = self._load_data(filename)
        self.m, self.n = feature_matrix.shape
        if self.feature_normalization:
            (feature_matrix,
             self.mu, self.sigma) = self._normalize_features(feature_matrix)
        # augment X by 1 (intercept feature):
        feature_matrix = np.hstack([np.ones((self.m, 1)), feature_matrix])
        # alternatively:
        # tmp = np.mat(np.ones((m, n + 1)))
        # tmp[:, 1:] = X
        # X = tmp
        self.cost = self.cost_type(feature_matrix, labels)

    def _load_data(self, filename):
        """The given filename contains the training set, one line per data point,
        consisting of comma separated features and label."""
        training_set = np.loadtxt(filename, delimiter=",", dtype=float)
        training_set = np.mat(training_set)
        # alternatively:
        # with open(filename, 'r') as datafile:
        #     training_set = np.vstack(np.mat(line.split(","), float)
        #                              for line in datafile)
        labels = training_set[:, -1] # last column
        feature_matrix = training_set[:, :-1] # all but last column
        return feature_matrix, labels

    def loadmat(self, filename):
        feature_matrix, labels = self._load_data(filename)
        self.m, self.n = feature_matrix.shape
        if self.feature_normalization:
            (feature_matrix,
             self.mu, self.sigma) = self._normalize_features(feature_matrix)
        # augment X by 1 (intercept feature):
        feature_matrix = np.hstack([np.ones((self.m, 1)), feature_matrix])
        # alternatively:
        # tmp = np.mat(np.ones((m, n + 1)))
        # tmp[:, 1:] = X
        # X = tmp
        self.cost = self.cost_type(feature_matrix, labels)

    def _load_data_mat(self, filename, name_feature_matrix="X", name_labels="y"):
        """The given filename contains the training set in matlab format"
        where the feature matrix is named consisting of comma separated features and label."""
        training_set = loadmat(filename)
        labels = np.mat(training_set["y"])
        feature_matrix = np.mat(training_set["X"]) # all but last column
        return feature_matrix, labels

class Regression(SupervisedLearner):
    pass

class Classification(SupervisedLearner):
    pass


class LogisticRegression(Classification):
    def __init__(self, max_iters, feature_normalization=False):
        self.cost_type = LogisticCost
        self.feature_normalization = feature_normalization

    def learn(self, theta_0=None):
        if theta_0==None:
            theta_0 = np.mat(np.zeros((self.n + 1, 1)))
        self.theta, J_history = scipy.optimize.optimize(self.cost, theta_0)
        return self.theta, J_history
        

class LinearRegression(Regression):
    def __init__(self, alpha, max_iters, feature_normalization=False):
        self.optimizer = GradDescent(alpha, max_iters)
        self.feature_normalization = feature_normalization
        self.cost_type = MeanSquaredError


    def learn(self, theta_0=None):
        if theta_0==None:
            theta_0 = np.mat(np.zeros((self.n + 1, 1)))
        self.theta, J_history = self.optimizer.optimize(self.cost, theta_0)
        return self.theta, J_history

    def _normalize_features(self, features, mu=None, sigma=None):
        if mu==None:
            mu = np.mean(features, 0)
        if sigma==None:
            sigma = np.std(features, 0)
        normalized = (features - mu) / sigma
        return normalized, mu, sigma

    def predict(self, data_point):
        if self.feature_normalization:
            data_point = self._normalize_features(data_point,
                                                  self.mu, self.sigma)[0]
        augmented_data_point = np.hstack([np.mat("1"), data_point])
        return augmented_data_point * self.theta

class NeuralNetwork(SupervisedLearner):
    def learn(__self__):
        raise NotImplementedError()

    def predict(__self__):
        raise NotImplementedError()


def main():
    pass
    
if __name__ == "__main__":
    main()

