import itertools
import collections
import time
import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils import check_array, check_random_state

from comprex.partition import Partition
from comprex.feature_set import FeatureSet
from comprex.code_table import CT

rng = np.random.RandomState(2018)

___author___ = 'HamedMP'


class CompreX(BaseEstimator, ClassifierMixin):
    def __init__(self,
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

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    # def get_params(self, deep=True):
    #     pass

    def transform(self, X, y=None):
        """
        Initializes parameters required for the algorithm.
        Sets input data `X`,  columns of `X` as `features, number of samples in the dataset.
        Creates the initial partition with Elementary feature sets.

        Parameters
        ----------
        :param X: Input data, preferred to be a pandas data frame with categorical data types.
        :param y: Optional, for sklearn compatibility
        :return: The initial partition, can be discarded
        """
        self.X = pd.DataFrame(X)
        self.y = y
        self.features = self.X.columns
        self.n = self.X.shape[0]  # Number of tuples TODO check
        self.m = self.X.shape[1]  # Number of features

        feature_set_tuples = [self.get_feature_name_domain_tuple(f) for f in self.X.columns]

        self.partition_init = Partition(feature_set_tuples=feature_set_tuples,
                                        X=self.X,
                                        n=self.n,
                                        logging_level=self.logging_level)

        return self.partition_init

    def fit(self, X, y=None):
        """
        Main fit method, looping through each feature set tuples and merge the ones with highest
        information gain.

        Parameters
        ----------
        :param X: Optional, for sklearn compatibility
        :param y: Optional, for sklearn compatibility
        :return: The final partition
        """

        self.X = pd.DataFrame(X)
        self.y = y
        self.features = self.X.columns
        self.n = self.X.shape[0]  # Number of tuples TODO check
        self.m = self.X.shape[1]  # Number of features

        feature_set_tuples = [self.get_feature_name_domain_tuple(f) for f in self.X.columns]

        self.partition_init = Partition(feature_set_tuples=feature_set_tuples,
                                        X=self.X,
                                        n=self.n,
                                        logging_level=self.logging_level)

        while True:
            mfs = self.merge_feature_sets(self.partition_init)
            self.partition_init, n_merges = self.merge_code_tables(self.partition_init, mfs)
            if n_merges == 0:
                break
            else:
                self.logger.debug(
                    'Anomaly score are \n{} in \n partition {}'.format(self.partition_init.find_anomalies(),
                                                                       self.partition_init))
            self.logger.debug('_____________________number of merges {}_____________________'.format(n_merges))

        return self.partition_init

    def predict(self, X, y=None):
        return self.partition_init.find_anomalies()

    def score(self, X, y=None):
        pass

    @staticmethod
    def _build_code_tables(partition):
        code_tables = [CT(feature_set) for feature_set in partition.get_feature_sets()]
        return code_tables

    def get_feature_name_domain_tuple(self, feature_name):
        select = feature_name[0] if type(feature_name) == list else [feature_name]

        grouped = self.X.groupby(feature_name)
        grouped_count = grouped[select].count()
        patterns = grouped_count.index
        usage = grouped_count.values
        return grouped, feature_name, patterns, usage

    # Compute IG between feature set tuples
    def merge_feature_sets(self, _partition):
        """
        Create unique combinations of feature sets, calculates the information gain
        and sort them in decreasing order.


        Parameters
        ----------
        :param _partition: The partition to merge its feature sets
        :return: Pandas data frame of merged feature sets in the decreasing order by
        their normalised information gain, with structure of columns=['F_i', 'F_j', 'F_ij', 'ig']
        """
        merged_feature_sets = []
        for i, j in itertools.combinations(_partition.get_feature_sets(), 2):
            temp_features = self.get_feature_name_domain_tuple(i.merge_with(j))
            temp_feature_set = FeatureSet(grouped=temp_features[0],
                                          features_list=temp_features[1],
                                          patterns_list=temp_features[2],
                                          features_usage_count=temp_features[3],
                                          X=self.X,
                                          n=self.n,
                                          temp=True)
            ig = i.calculate_IG(j, temp_feature_set)

            merged_feature_sets.append((i, j, temp_feature_set, ig))

        merged_feature_sets = pd.DataFrame(merged_feature_sets, columns=['F_i', 'F_j', 'F_ij', 'ig'])

        merged_feature_sets.sort_values(by='ig', ascending=False, inplace=True)

        return merged_feature_sets

    # Loop through Feature Set tuples in decreasing normalised IG
    def merge_code_tables(self, _partition_init, merged_feature_sets=None):
        number_of_merges = 0

        # In case of discarding the merge
        p_copy = _partition_init

        if merged_feature_sets.empty:
            return _partition_init, number_of_merges
        else:
            self.logger.debug('Merged feature sets are: \n {}'.format(merged_feature_sets))
        # Loop through merged feature sets in decreasing normalised IG to find FSs to combine.
        for i in range(merged_feature_sets.shape[0]):  # L(5)

            cost_init = _partition_init.calculate_total_cost(self.X)  # L(6)

            f_i = merged_feature_sets.iloc[i]['F_i']
            f_j = merged_feature_sets.iloc[i]['F_j']
            f_ij = merged_feature_sets.iloc[i]['F_ij']
            ig = merged_feature_sets.iloc[i]['ig']

            # if ig == 0:
            #     break

            self.logger.info('\n\n\n+++++++ F_ij: {} +++++++'.format(f_ij))

            ct_i = f_i.CT
            ct_j = f_j.CT

            c_hat_ij = CT.merge_CTs(ct_i, ct_j)

            unique_rows = CT.unique_rows(ct_i, ct_j)

            # L(10)
            f_ij.add_code_table(self.X, c_hat_ij)

            # L(9), p_hat
            new_partition_fss = _partition_init.get_new_feature_sets_list_by_add_remove(add=f_ij,
                                                                                        remove=[f_i, f_j])
            new_partition = Partition(feature_set_list=new_partition_fss,
                                      X=self.X,
                                      n=self.n,
                                      logging_level=self.logging_level)

            # Calculate new rows of C_i_j before appending U to it.
            for unique_row in unique_rows:  # L(12)
                if unique_row[0] in list(c_hat_ij.index) and unique_row[1] in list(c_hat_ij.index):
                    self.logger.debug('old c_hat_ij is \n{}'.format(c_hat_ij))
                    c_hat_ij = CT.update_c_hat(c_hat_ij, unique_row)
                    self.logger.debug('new c_hat_ij is \n{}'.format(c_hat_ij))
                    # Add new_pattern to FS and CT
                    new_partition.update_feature_set(f_ij.name, self.X, c_hat_ij)
                    self.logger.info('Inner loop f_ij: {} and row: {}'.format(f_ij, unique_row))
                    self.logger.debug('C for feature set f_ij {} is \n{}'.format(f_ij, f_ij.CT.C))

                    new_cost= new_partition.calculate_total_cost(self.X)
                    old_cost = _partition_init.calculate_total_cost(self.X)

                    self.logger.debug('Old partition was:\n{}\nAnd new Partitions is\n{}'.format(
                        [fsp.patterns_list for fsp in _partition_init.feature_sets],
                        [fsp.patterns_list for fsp in new_partition.feature_sets]
                    ))

                    if new_cost < old_cost:
                        self.logger.debug('New partition cost was lower by: {} '.format(old_cost - new_cost))
                        _partition_init = new_partition
                        # print(c_hat_ij)
                    else:
                        self.logger.debug('----No change to partition----\n'.format())

            if _partition_init.calculate_total_cost(self.X) < cost_init:
                number_of_merges += 1
                self.logger.info('Number of merges is: {}'.format(number_of_merges))
                return _partition_init, number_of_merges
            else:
                self.logger.info('Merge is rejected and total number of merges is: {}'.format(number_of_merges))
                _partition_init = p_copy

                self.logger.info('============================================='.format())
        return _partition_init, number_of_merges
