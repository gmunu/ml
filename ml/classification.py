from learner import *


class Classification(SupervisedLearner):
    pass


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
        

