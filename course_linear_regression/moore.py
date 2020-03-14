import re
import numpy as np
import matplotlib.pyplot as plt
import os.path

X = []
Y  = []

non_decimal = re.compile(r'[^\d]+')

data_folder = os.path.dirname('/home/talg/udemy_deep_learning/machine_learning_examples/linear_regression_class/')
data_file = 'moore.csv'

data_path = os.path.join(data_folder, data_file)

for line in open(data_path):
    r = line.split('\t')
    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))

    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# caclulate a and b

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator
Yhat = a * X + b
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# calculated R-sqrd
d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squired = ", r2)



