import itertools
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

rng = np.random.RandomState(2018)


def join_tuples(t1, t2):
    if type(t1) == tuple and type(t2) == tuple:
        return t1 + t2
    elif type(t1) is tuple:
        t = *t1, t2
        return t
    elif type(t2) is tuple:
        t = t1, *t2
        return t
    else:
        return t1, t2


class CT:
    def __init__(self,
                 feature_set,
                 X=None,
                 C=None,  # For creating new partition and skipping the creation of C right away!
                 logging_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logging_level = logging_level
        self.logger.setLevel(self.logging_level)

        if not self.logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                          datefmt='%m/%d/%Y %I:%M:%S %p')
            sh = logging.StreamHandler()
            sh.setLevel(logging_level)
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

        self.X = X
        self.feature_set = feature_set
        self.patterns = np.array(self.feature_set.get_patterns())
        if C is None:
            self.C = self.calculate_C(X)
        else:
            self.C = C

    def __str__(self):
        return 'CT' + str(self.feature_set)[2:]

        # '__'.join(self.feature_set) if type(self.features) == list else self.features

    def __repr__(self):
        return str(self)

    def calculate_C(self, X):
        one_hot = MultiLabelBinarizer(classes=np.arange(len(self.X)))
        appearance = [self.feature_set.get_features_appearance_in(pattern) for pattern in self.patterns]
        self.logger.debug('Patterns in the CT are: {},  '
                          'Features appearances are:{}'.format([pattern for pattern in self.patterns], appearance))

        return one_hot.fit_transform(appearance)  # Sparse matrix C

    def update_C(self, new_C):
        self.C = new_C

    def update_patterns(self, new_patterns):
        self.patterns = new_patterns

    def calculate_anomaly_score(self):
        return np.sum(self.C)

    def calculate_data_encoding_cost(self):  # L(D | CT)
        """
        Row sum of C is # of usage of each pattern in D. When multiplied by its respective code
        length and summed, we get the length of encoding all data in this CT.
        """
        if self.C is None:
            raise ValueError('Please call `.calculate_C()` before this method.')
        self.C = self.C[np.sum(self.C, axis=1) > 0]
        data_cost = np.sum(np.sum(self.C, axis=1) * self.calculate_patterns_length())

        self.logger.debug('Data cost L(D | CT) is {} for CT = {}'.format(data_cost, self))
        return data_cost

    @staticmethod
    def calculate_external_patterns_length(ex_C):  # L(code(p) | CT)
        return - np.log2(np.sum(ex_C.values, axis=1) / np.sum(ex_C.values))

    def calculate_model_cost(self):  # L(CT)
        ct_cost = np.sum(self._patterns_encoding_cost() + self.calculate_patterns_length())
        self.logger.debug('L(CT) is {} for CT = {}'.format(ct_cost, self))
        return ct_cost

    def calculate_patterns_length(self):  # L(code(p) | CT)
        return - np.log2(np.sum(self.C, axis=1) / np.sum(self.C))

    def _patterns_encoding_cost(self):  # sum - r_i log(p_i)
        singleton_counts_array = np.array(list(Counter(itertools.chain(self.patterns)).values()))
        self.logger.debug('Singleton count for CT {} is {}'.format(self, singleton_counts_array))
        patterns_cost = np.sum(
            -1 * singleton_counts_array * np.log2(singleton_counts_array / np.sum(singleton_counts_array)))
        self.logger.debug('-r_i log(p_i) Pattern encoding cost for {} with patterns {} is {}'
                          .format(self, self.patterns, patterns_cost))
        return patterns_cost

    @staticmethod
    def merge_CTs(ct_i, ct_j):
        c_hat_ij = pd.DataFrame(np.concatenate((ct_i.C, ct_j.C)),
                                index=np.concatenate((ct_i.patterns, ct_j.patterns)))

        # Find code length before any modification on the C_hat matrix
        code_len = CT.calculate_external_patterns_length(c_hat_ij)
        c_hat_ij = c_hat_ij.assign(usage=c_hat_ij.sum(axis=1))
        c_hat_ij = c_hat_ij.assign(length=code_len).sort_values(
            by=['length', 'usage'],
            ascending=False)
        c_hat_ij = c_hat_ij[c_hat_ij['usage'] > 0].iloc[:, :-2]  # L(8)

        return c_hat_ij

    @staticmethod
    def update_c_hat(c_hat_ij, unique_row):
        new_pattern = join_tuples(unique_row[0], unique_row[1])
        # self.logger.info('\n+++++++ new patterns: {} +++++++'.format(new_pattern))

        app = pd.Series(*np.logical_and(c_hat_ij.loc[[unique_row[0]]],
                                        c_hat_ij.loc[[unique_row[1]]])
                        .values.astype(np.float)
                        .tolist(),
                        name=new_pattern)

        # Reduce number of usages (L14)
        c_hat_ij.loc[[unique_row[0]]] = c_hat_ij.loc[[unique_row[0]]] - app
        c_hat_ij.loc[[unique_row[1]]] = c_hat_ij.loc[[unique_row[1]]] - app
        # Remove patterns with 0 usage (L15)
        # Additionally check if it will not come up in the next loops
        c_hat_ij = c_hat_ij[np.sum(c_hat_ij, axis=1) > 0]
        c_hat_ij = c_hat_ij.append(app)  # L(13)

        return c_hat_ij

    @staticmethod
    def unique_rows(ct_i, ct_j):
        u_matrix = pd.DataFrame(np.dot(ct_i.C, ct_j.C.T),  # L(11)
                                columns=ct_j.patterns,
                                index=ct_i.patterns)
        # [i[2] for i in merged_U_patterns(ct_1.patterns, ct_2.patterns)])
        return list(u_matrix.unstack().sort_values(ascending=False).index.values)
