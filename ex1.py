import numpy as np

filename_univariate_data = 'ex1data1.txt'
filename_multivariate_data = 'ex1data2.txt'

def load_data(filename):
    """The given filename contains the training set, one line per data point,
    consisting of comma separated features and label."""
    training_set = np.loadtxt(filename, delimiter=",", dtype=float)
    training_set = np.mat(training_set)
    # alternatively:
    # with open(filename, 'r') as datafile:
    #     training_set = np.vstack(np.mat(line.split(","), float)
    #                              for line in datafile)
    labels = training_set[:,-1] # last column
    feature_matrix = training_set[:,:-1] # all but last column
    return feature_matrix, labels

def feature_normalization(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X - mu) / sigma
    return X, mu, sigma

def compute_cost(X, y, theta):
    m = len(y)
    h = X * theta # hypothesis
    J = (h - y).T * (h - y) / (2 * m)
    return J

def grad_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters + 1, 1))
    J_history[0] = compute_cost(X, y, theta)
    for i in xrange(num_iters):
        h = X * theta
        theta = theta - alpha / m * X.T * (h - y)
        J_history[i + 1] = compute_cost(X, y, theta)
    return theta, J_history

class LinearRegression:
    def __init__(self, alpha, num_iters, normalize_features=False):
        self.alpha = alpha
        self.num_iters = num_iters
        self.normalize_features = normalize_features

    def load(self, filename):
        self.feature_matrix, self.labels = load_data(filename)
        self.m, self.n = self.feature_matrix.shape
        if self.normalize_features:
            (self.feature_matrix,
             self.mu, self.sigma) = feature_normalization(self.feature_matrix)

    def learn(self):
        # augment X by 1 (intercept feature):
        self.feature_matrix = np.hstack([np.ones((self.m, 1)),
                                         self.feature_matrix])
        # alternatively:
        # tmp = np.mat(np.ones((m, n + 1)))
        # tmp[:, 1:] = X
        # X = tmp
        self.theta = np.mat(np.zeros((self.n + 1, 1)))
        self.theta, J_history = grad_descent(self.feature_matrix,
                                             self.labels, self.theta,
                                             self.alpha, self.num_iters)
        return self.theta, J_history

    def _normalize_data_point(self, data_point):
        return (data_point - self.mu) / self.sigma

    def predict(self, data_point):
        if self.normalize_features:
            data_point = self._normalize_data_point(data_point)
        augmented_data_point = np.hstack([np.mat("1"), data_point])
        return augmented_data_point * self.theta

def main():
    print "Univariate Linear Regression:"
    lr = LinearRegression(alpha=0.01, num_iters=1500)

    lr.load(filename_univariate_data)
    theta, J_history = lr.learn()
    print "initial cost J: " + str(J_history[0])
    print str(J_history[1:])
    print "final thetas: " + str(theta)
    
    print "Miltivariate Linear Regression:"
    lr = LinearRegression(alpha=0.01, num_iters=400, normalize_features=True)
    lr.load(filename_multivariate_data)
    theta, J_history = lr.learn()
    price = lr.predict(np.mat("1650 3"))
    print "initial cost J: " + str(J_history[0])
    print str(J_history[1:])
    print "final thetas: " + str(theta)
    print "predicted price for 1650 ft^2, 3br: $" + str(price)

if __name__ == "__main__":
    main()

