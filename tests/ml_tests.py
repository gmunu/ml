from ml.ml import *

filename_univariate_linear_data = 'tests/ex1data1.txt'
filename_multivariate_linear_data = 'tests/ex1data2.txt'
filename_logistic_data = 'tests/ex2data1.txt'
filename_neural_network = 'tests/ex4data1.mat'

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
