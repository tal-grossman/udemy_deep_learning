import numpy as np
import  matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

# center the 50 first point at (-2, -2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))

# center the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# the first 50 will be set to 0 and the last 50 will be set to 1
T = np.array([0] * 50 + [1] * 50)

# add column of one
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


Y = sigmoid(z)


# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


print("cross_entropy(T, Y): ", cross_entropy(T, Y))


# try it with our closed-form solution
w = np.array([0, 4, 4])

# calculate the model output
z = Xb.dot(w)
Y = sigmoid(z)

# calculate the cross-entropy error
print(cross_entropy(T, Y))


# y = -x
plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)

x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)

plt.show()