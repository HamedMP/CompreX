import numpy as np
import pandas as pd

from comprex import CompreX

from sklearn.utils.estimator_checks import check_estimator

check_estimator(CompreX)

# rng = np.random.RandomState(2018)
#
# X = pd.DataFrame([['a', 'b', 'x', 'n'],
#                   ['a', 'c', 'x', 'n'],
#                   ['a', 'c', 'x', 'n'],
#                   ['a', 'c', 'x', 'n'],
#                   ['a', 'c', 'y', 'm']],
#                  columns=['f1', 'f2', 'f3', 'f4'],
#                  index=[i for i in np.arange(5)],
#                  dtype='category')
#
# estimator = CompreX()
# estimator.transform(X)
# estimator.fit(X)
# print(estimator.predict())
