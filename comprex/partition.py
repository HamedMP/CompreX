import logging
import numpy as np
from comprex.feature_set import FeatureSet

rng = np.random.RandomState(2018)


class Partition:
    def __init__(self,
                 X,
                 n,
                 feature_set_list=None,
                 feature_set_tuples=None,
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

        if feature_set_tuples is not None and len(feature_set_tuples) > 0:
            self.feature_set_tuples = feature_set_tuples
            self.feature_sets = \
                [FeatureSet(grouped=feature[0],
                            features_list=feature[1],
                            patterns_list=feature[2],
                            features_usage_count=feature[3],
                            X=X,
                            n=n,
                            logging_level=logging_level) for feature in self.feature_set_tuples]
        elif feature_set_list is not None and len(feature_set_list) > 0:
            self.feature_sets = feature_set_list
        else:
            raise ValueError('You should provide either feature_set_tuples or feature_sets_list')

        self.code_tables = self.build_code_tables(X)
        self.IG = None

    def __str__(self):
        return 'P__' + (str([fs.name for fs in self.feature_sets]))

    def __repr__(self):
        return str(self)

    def calculate_partition_cost(self, m):  # L(P)
        k = len(self.feature_sets)
        partition_cost = self.log_star(k) + m * np.log2(k)
        self.logger.info('Partition cost is {}'.format(partition_cost))
        return partition_cost

    def log_star(self, k):
        x = np.log2(k)
        x_sum = x
        while x > 0 and np.log2(x) > 0:
            x = np.log2(x)
            x_sum += x

        return x_sum

    def find_anomalies(self):
        cost_per_ct = [ct.calculate_anomaly_score() for ct in self.code_tables]
        return sum(cost_per_ct)

    def calculate_total_cost(self, X):  # L(P, CT, D)
        data_cost = np.sum([ct.calculate_data_encoding_cost() for ct in self.code_tables])
        model_cost = np.sum([ct.calculate_model_cost() for ct in self.code_tables])

        ct_costs = model_cost + data_cost
        self.logger.info('_________Current partition is {}__________'.format(self.feature_sets))
        [self.logger.debug('Partition C\'s for fs {} are \n{}'.format(fs, fs.CT.C)) for fs in self.feature_sets]
        self.logger.info('Partition\'s CT costs are: data L(D | CT) {} and model {} with total CT cost of {}'
                          .format(data_cost, model_cost, ct_costs))
        total_cost = self.calculate_partition_cost(X.shape[1]) + ct_costs
        self.logger.info('L(P, CT, D) Total partition cost is: {}'.format(total_cost))
        return total_cost

    def build_code_tables(self, X):
        return [fs.CT for fs in self.feature_sets]

    def get_feature_sets(self):
        return self.feature_sets

    def get_feature_set(self, name):
        for fs in self.feature_sets:
            if fs.name == name:
                return fs

    def update_feature_set(self, fs_name, X, c_hat_ij):
        fs = self.get_feature_set(fs_name)
        fs.add_code_table(X, c_hat_ij)
        self.build_code_tables(X)

    def calculate_information_gain(self):  # IG
        pass

    def get_new_feature_sets_list_by_add_remove(self, add, remove):
        if add is None:
            raise ValueError('You should provide a FeatureSet to add to current feature sets')

        new_feature_sets = [add]
        if len(remove) > 0:
            for fs in self.feature_sets:
                if fs not in remove:
                    new_feature_sets.append(fs)

        return new_feature_sets
