"""
An example usage with dummy data set

X is the dummy dataset and we run CompreX algorithm on it by creating an estimator and
running transform and fit  method of it. At the end you can call predict() method to
get anomaly score for each sample in the dataset.
"""

import numpy as np
import pandas as pd
import logging

from comprex.comprex import CompreX

from sklearn.utils.estimator_checks import check_estimator

# check_estimator(CompreX)

rng = np.random.RandomState(2018)

X = pd.DataFrame([['a', 'b', 'x'],
                  ['a', 'b', 'x'],
                  ['a', 'b', 'x'],
                  ['a', 'b', 'x'],
                  ['a', 'c', 'x'],
                  ['a', 'c', 'y']],
                 columns=['f1', 'f2', 'f3'],
                 index=[i for i in np.arange(6)],
                 dtype='category')

estimator = CompreX(logging_level=logging.DEBUG)
estimator.transform(X)
estimator.fit(X)
print(estimator.predict(X))
