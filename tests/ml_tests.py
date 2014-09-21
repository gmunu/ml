from nose import with_setup
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal
import numpy as npy
from ml.ml import *

filename_univariate_linear_data = 'tests/ex1data1.txt'
filename_multivariate_linear_data = 'tests/ex1data2.txt'
filename_logistic_data = 'tests/ex2data1.txt'
filename_neural_network = 'tests/ex4data1.mat'

class TestUnivariateLinearRegression:

    def setup(self):
        self.lr = LinearRegression(alpha=0.01, max_iters=1500)
        self.lr.load(filename_univariate_linear_data)

    def teardown(self):
        self.lr = None

    def test_univariate_linear_regression_cost(self):
        theta, J_history = self.lr.learn()
        actual = J_history[0,0]
        expected = 32.07273
        assert_almost_equal(actual, expected, places=4)

    def test_univariate_linear_regression_gradient_descent(self):
        theta, J_history = self.lr.learn()
        # final cost:
        J_min = J_history[-1, 0]
        desired_J_min = 4.483388
        assert_almost_equal(J_min, desired_J_min, places=5)
        # final thetas:
        desired_theta = npy.mat("-3.63029; 1.16636")
        assert_array_almost_equal(theta, desired_theta, decimal=5)

#@with_setup(setup_univariate_linear_regression, teardown_univariate_linear_regression)

def test_ex1():
    print "Univariate Linear Regression:"
    lr = LinearRegression(alpha=0.01, max_iters=1500)
    lr.load(filename_univariate_linear_data)
    theta, J_history = lr.learn()
    print "initial cost J: " + str(J_history[0])
    print str(J_history[1:])
    print "final thetas: " + str(theta)

def test_ex1_extra():
    print "Miltivariate Linear Regression:"
    lr = LinearRegression(alpha=0.01, max_iters=400, 
                          feature_normalization=True)
    lr.load(filename_multivariate_linear_data)
    theta, J_history = lr.learn()
    price = lr.predict(np.mat("1650 3"))
    print "initial cost J: " + str(J_history[0])
    print str(J_history[1:])
    print "final thetas: " + str(theta)
    print "predicted price for 1650 ft^2, 3br: $" + str(price)

def test_ex2():
    print "Logistic Regression:"
    lr = LogisticRegression(alpha=0.01, max_iters=400)
    lr.load(filename_logistic_data)
    theta, J_history = lr.learn()
    price = lr.predict(np.mat("1650 3"))
    print "initial cost J: " + str(J_history[0])
    print str(J_history[1:])
    print "final thetas: " + str(theta)
    print "predicted price for 1650 ft^2, 3br: $" + str(price)

def test_ex4():
    print "Neural Network:"
    nn = NeuralNetwork()
    nn.loadmat(filename_neural_network)
