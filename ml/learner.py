import numpy as npy

from dataset import *
from cost import *
from optimization import *

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



