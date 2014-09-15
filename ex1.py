import numpy as np

filename_univariate_data = 'ex1data1.txt'
filename_multivariate_data = 'ex1data2.txt'

def load_data(filename):
    """The given filename contains the training set, one line per data point,
    consisting of comma separated features and label."""
    training_set = np.loadtxt(filename, delimiter=",", dtype=float)
    training_set = np.mat(training_set)
    # or:
    # with open(filename, 'r') as datafile:
    #     training_set = np.vstack(np.mat(line.split(","), float)
    #                              for line in datafile)
    labels = training_set[:,-1] # last column
    feature_matrix = training_set[:,:-1] # all but last column
    return feature_matrix, labels

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

def linear_regression(alpha, num_iters, filename):
    X, y = load_data(filename)
    m, n = X.shape
    # augment X by 1 (intercept feature):
    X = np.hstack([np.ones((m,1)), X])
    # or:
    # tmp = np.mat(np.ones((m, n + 1)))
    # tmp[:, 1:] = X
    # X = tmp
    theta = np.mat(np.zeros((n + 1, 1)))
    return grad_descent(X, y, theta, alpha, num_iters)

def univariate_linear_regression(alpha, num_iters):
    return linear_regression(alpha, num_iters, filename_univariate_data)

def multivariate_linear_regression(alpha, num_iters):
    return linear_regression(alpha, num_iters, filename_multivariate_data)

def main():
    print "Univariate Linear Regression:"
    theta, J_history = univariate_linear_regression(alpha=0.01, num_iters=1500)
    print "initial cost J: " + str(J_history[0])
    print str(J_history[1:])
    print "final thetas: " + str(theta)


if __name__ == "__main__":
    main()

