import numpy as np
import  matplotlib.pyplot as plt

import os.path

data_folder = os.path.dirname('/home/talg/udemy_deep_learning/machine_learning_examples/linear_regression_class/')
data_file = 'data_poly.csv'

data_path = os.path.join(data_folder, data_file)

# load the data
X = []
Y = []

for line in open(data_path):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))


X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:, 1], Y)
plt.show()

# calculate weiights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# plot is all together
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()