import logging
from comprex.code_table import CT
import numpy as np

rng = np.random.RandomState(2018)


class FeatureSet:
    def __init__(self,
                 features_list,
                 patterns_list,
                 features_usage_count,
                 n,
                 X=None,
                 grouped=None,
                 temp=False,  # If True, it will create CT and C by data, with False it will get C as param.
                 logging_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logging_level = logging_level
        self.logger.setLevel(logging_level)

        if not self.logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                          datefmt='%m/%d/%Y %I:%M:%S %p')
            sh = logging.StreamHandler()
            sh.setLevel(logging_level)
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

        # Name of the features (columns) in this F.
        self.features = features_list if type(features_list) == list else [features_list]

        # Domain of a feature, a.k.a. unique values it takes, a.k.a. patterns
        self.patterns_list = patterns_list  # if type(patterns_list) == list else [patterns_list]
        self.patterns_usage_count = features_usage_count
        self.grouped = grouped
        self.cardinality = len(features_list) if type(features_list) == list else 1
        self.n = n

        if X is not None and temp is False:
            self.X = X
            self.CT = self.build_code_table(X)
        self.name = self.get_name()

    def get_name(self):
        return str(self)

    def __str__(self):
        return 'FS__' + '__'.join(self.features)  # if type(self.features) == list else self.features

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    def get_features(self):
        return self.features

    def get_patterns(self):
        return self.patterns_list

    def get_patterns_usage_count(self):
        return self.patterns_usage_count

    def get_features_appearance_in(self, pattern):
        return self.grouped.indices[pattern].tolist()

    #         return self.grouped.indices[tuple(pattern)]

    def get_features_appearance_all(self):
        print('patterns = ', [pattern for pattern in self.patterns_list])
        return [self.grouped.indices[tuple(pattern)] for pattern in self.patterns_list]

    def calculate_entropy(self):  # H(F)
        entropy = - np.sum(self.patterns_usage_count / self.n * np.log2(self.patterns_usage_count / self.n))
        self.logger.debug('Entropy is {} for featureset {}'.format(entropy, self.get_name()))
        return entropy

    def calculate_IG(self, f_j, temp_feature_set):
        ig = (self.calculate_entropy()
              + f_j.calculate_entropy()
              - temp_feature_set.calculate_entropy()) / temp_feature_set.cardinality

        return ig

    def merge_with(self, other_feature_set):
        return self.features + other_feature_set.get_features()

    def build_code_table(self, X):  # When it's final CT
        self.CT = CT(self, X=X, logging_level=self.logging_level)
        return self.CT

    def add_code_table(self, X, C_hat_ij):
        self.CT = CT(feature_set=self, X=X, C=C_hat_ij)
