import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os.path

data_folder = os.path.dirname('/home/talg/udemy_deep_learning/machine_learning_examples/linear_regression_class/')
data_file = 'data_2d.csv'

data_path = os.path.join(data_folder, data_file)

# load the data
X = []
Y = []

for line in open(data_path):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])
    Y.append(float(y))


X = np.array(X)
Y = np.array(Y)

# plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0], X[:,1], Y)
# plt.show()

# calculate weights of model

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# cumpute r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared = ", r2)