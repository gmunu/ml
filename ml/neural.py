from learner import *

class NeuralNetwork(SupervisedLearner):

    def __init__(self, alpha, max_iters, feature_normalization=False):
        self.optimizer = GradDescent(alpha, max_iters)
        self.feature_normalization = feature_normalization
        self.cost_type = NeuralNetworkCost

    def learn(self, theta_0=None):
        raise NotImplementedError()

    def predict(self, data_point):
        raise NotImplementedError()



