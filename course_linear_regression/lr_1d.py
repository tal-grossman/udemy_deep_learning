import numpy as np
import matplotlib.pyplot as plt

import os.path

data_folder = os.path.dirname('/home/talg/udemy_deep_learning/machine_learning_examples/linear_regression_class/')
data_file = 'data_1d.csv'

data_path = os.path.join(data_folder, data_file)

# load the data
X = []
Y = []

for line in open(data_path):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# X and Y to numpy arrays

X = np.array(X)
Y = np.array(Y)

# plot

plt.scatter(X, Y)
plt.show()

# caclulate a and b

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# calculate predicted Y
Yhat = a * X + b
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# calculated R-sqrd
d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squired = ", r2)
