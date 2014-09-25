# from nose import with_setup
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal, assert_allclose
import numpy as npy
import scipy.io as sio
from ml.ml import *

filename_univariate_linear_data = 'tests/ex1data1.txt'
filename_multivariate_linear_data = 'tests/ex1data2.txt'
filename_logistic_data = 'tests/ex2data1.txt'
filename_neural_network = 'tests/ex4data1.mat'

class TestSupervisedLoad:

    filename_multivariate_csv_data = 'tests/ex1data2.txt'
    filename_multivariate_mat_data = 'tests/ex1data2.mat'

    def setup(self):
        self.sl = SupervisedLearner(alpha=0.01, max_iters=40)

    def teardown(self):
        self.sl = None

    def test_csv_load(self):
        (feature_matrix,
         labels) = self.sl.load(self.filename_multivariate_csv_data)
        some_features = feature_matrix[-3:,:]
        expected_features = npy.mat("1 852 2; 1 1852 4; 1 1203 3")
        assert_array_almost_equal(some_features, expected_features)
        some_labels = labels[-3:, 0]
        expected_labels = npy.mat("179900; 299900; 239500")
        assert_array_almost_equal(some_labels, expected_labels)

    def test_mat_load(self):
        (feature_matrix,
         labels) = self.sl.loadmat(self.filename_multivariate_mat_data)
        some_features = feature_matrix[-3:,:]
        expected_features = npy.mat("1 852 2; 1 1852 4; 1 1203 3")
        assert_array_almost_equal(some_features, expected_features)
        some_labels = labels[-3:, 0]
        expected_labels = npy.mat("179900; 299900; 239500")
        assert_array_almost_equal(some_labels, expected_labels)


class TestUnivariateLinearRegression:

    def setup(self):
        self.lr = LinearRegression(alpha=0.01, max_iters=1500)
        self.lr.load(filename_univariate_linear_data)

    def teardown(self):
        self.lr = None

    def test_cost(self):
        theta, J_history = self.lr.learn()
        initial_cost = J_history[0,0]
        expected_initial_cost = 32.07273
        assert_almost_equal(initial_cost, expected_initial_cost, places=5)

    def test_gradient_descent(self):
        theta, J_history = self.lr.learn()
        # final cost:
        J_min = J_history[-1, 0]
        expected_J_min = 4.483388
        assert_almost_equal(J_min, expected_J_min, places=5)
        # final thetas:
        expected_theta = npy.mat("-3.63029; 1.16636")
        assert_array_almost_equal(theta, expected_theta, decimal=5)

    def test_predict(self):
        self.lr.learn()
        prediction = self.lr.predict(npy.mat("3.5"))
        expected_prediction = 0.4519767868
        assert_almost_equal(prediction, expected_prediction, places=5)
        prediction = self.lr.predict(npy.mat("7"))
        expected_prediction = 4.5342450129
        assert_almost_equal(prediction, expected_prediction, places=5)


class TestMultivariateLinearRegression:

    def setup(self):
        self.lr = LinearRegression(alpha=0.01, max_iters=400,
                                   feature_normalization=True)
        self.lr.load(filename_multivariate_linear_data)

    def teardown(self):
        self.lr = None

    def test_cost(self):
        theta, J_history = self.lr.learn()
        initial_cost = J_history[0,0]
        expected_initial_cost = 6.55915481e10 # value from octave
        assert_allclose(initial_cost, expected_initial_cost, rtol=0.01)

    def test_gradient_descent_first_step(self):
        theta, J_history = self.lr.learn()
        initial_cost = J_history[1,0]
        expected_initial_cost = 6.430074959e10
        assert_allclose(initial_cost, expected_initial_cost, rtol=0.01)

    def test_gradient_descent_final_cost(self):
        theta, J_history = self.lr.learn()
        J_min = J_history[-1, 0]
        expected_J_min = 2.10885006e9 # value from octave
        assert_allclose(J_min, expected_J_min, rtol=0.01)

    def test_gradient_descent_final_thetas(self):
        theta, J_history = self.lr.learn()
        expected_theta = npy.mat("334302.063993; 100087.116006; 3673.548451")
        assert_allclose(theta, expected_theta, rtol=0.2)

    def test_predict(self):
        self.lr.learn()
        prediction = self.lr.predict(npy.mat("1650 3"))
        expected_prediction = 289314.620338
        assert_allclose(prediction, expected_prediction, rtol=0.01)


class TestNeuralNetwork:

    filename_training_set = 'tests/ex4data1.mat'
    filename_weights = 'tests/ex4weights.mat'

    def setup(self):
        self.nn = NeuralNetwork(alpha=0.01, max_iters=40)
        self.nn.loadmat(TestNeuralNetwork.filename_training_set)
        data = sio.loadmat(TestNeuralNetwork.filename_weights)
        self.thetas = [npy.mat(data["Theta1"]), npy.mat(data["Theta2"])]

    def teardown(self):
        self.nn = None

    def test_cost(self):
        cost = self.nn.cost.compute(self.thetas)
        expected_cost = 0.2876291652 # value from octave
        assert_allclose(cost, expected_cost, rtol=0.01)

    def test_regularized_cost(self):
        cost = self.nn.cost.compute(self.thetas, regularization_const=1)
        expected_cost = 0.383769859 # value from octave
        assert_allclose(cost, expected_cost, rtol=0.01)

    def test_gradient_descent_first_step(self):
        theta, J_history = self.nn.learn()
        initial_cost = J_history[1,0]
        expected_initial_cost = 6.430074959e10
        assert_allclose(initial_cost, expected_initial_cost, rtol=0.01)


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
    price = lr.predict(npy.mat("1650 3"))
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
    theta, J_history = nn.learn()
    print "initial cost J: " + str(J_history[0])
    print str(J_history[1:])
    print "final thetas: " + str(theta)


