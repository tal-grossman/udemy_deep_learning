import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os.path

data_folder = os.path.dirname('/home/talg/udemy_deep_learning/course_linear_regression/data/')
data_file = 'mlr02.xls'

data_path = os.path.join(data_folder, data_file)

df = pd.read_excel(data_path)

X = df.as_matrix()

plt.scatter(X[:, 1], X[:, 0])
# plt.show()

plt.scatter(X[:, 2], X[:, 0])
# plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]

X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]


def get_w(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    return w


def get_Yhat(X, w):
    return X.dot(w)


def get_r2(X, Y):
    d1 = Y - get_Yhat(X, get_w(X, Y))
    d2 = Y - Y.mean()

    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2


print("r2 to for X2 only: ", get_r2(X2only, Y))
print("r2 to for X3 only: ", get_r2(X3only, Y))
print("r2 to for both: ", get_r2(X, Y))
