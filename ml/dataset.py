import numpy as npy

class Dataset(object):

    def __init__(self):
        self.is_normalized = False

    def normalize_features(self):
        if self.feature_normalization:
            (self.feature_matrix, self.mu,
             self.sigma) = self._normalize_features(self.feature_matrix)
            self.is_normalized = True
        return self

    @staticmethod
    def _normalize_features(features, mu=None, sigma=None):
        if mu==None:
            mu = npy.mean(features, 0)
        if sigma==None:
            sigma = npy.std(features, 0)
        normalized_features = (features - mu) / sigma
        return normalized_features, mu, sigma


class SupervisedDataset(Dataset):
    
    @classmethod
    def from_file(cls, filename, data_format="csv",
                 feature_normalization=False):
        dataset = cls()
        if data_format == "csv":
            feature_matrix, labels = cls._load_csv(filename)
        elif data_format == "mat":
            feature_matrix, labels = cls._load_mat(filename)
        m, n = feature_matrix.shape
        mu, sigma = None, None
        if feature_normalization:
            (feature_matrix,
             mu, sigma) = cls._normalize_features(feature_matrix)
        # augment X by 1 (intercept feature):
        feature_matrix = npy.hstack([npy.ones((m, 1)), 
                                          feature_matrix])
        # alternatively:
        # tmp = npy.mat(npy.ones((m, n + 1)))
        # tmp[:, 1:] = X
        # X = tmp
        dataset = cls()
        dataset.feature_matrix = feature_matrix
        dataset.labels = labels
        dataset.m, dataset.n = m, n
        dataset.features_are_normalized = feature_normalization
        dataset.mu = mu
        dataset.sigma = sigma
        return dataset

    @staticmethod
    def _load_csv(filename, 
                  # by default, the labels are in the last column:
                  label_column=-1):
        """The given filename contains the training set, one line per data 
        point, consisting of comma separated features and label in column
        label_column (last column by default)."""
        training_set = npy.loadtxt(filename, delimiter=",", dtype=float)
        training_set = npy.mat(training_set)
        # alternatively:
        # with open(filename, 'r') as datafile:
        #     training_set = npy.vstack(npy.mat(line.split(","), float)
        #                              for line in datafile)
        labels = training_set[:, label_column] 
        # delete label column:
        feature_matrix = npy.delete(training_set, label_column, 1)
        return feature_matrix, labels

    @staticmethod
    def _load_mat(filename, name_feature_matrix="X", name_labels="y"):
        """The given filename contains the training set in matlab format,
        where the feature matrix and the labels are named name_feature_matrix 
        and named name_labels ("X" and "y" by default), respectively."""
        import scipy.io as sio
        training_set = sio.loadmat(filename)
        labels = npy.mat(training_set[name_labels])
        feature_matrix = npy.mat(training_set[name_feature_matrix])
        return feature_matrix, labels



