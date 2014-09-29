from learner import *


class Regression(SupervisedLearner):
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


