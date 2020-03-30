import numpy as np
from process import get_binary_data

X, Y, _, _ = get_binary_data()

D = X.shape[1]
W = np.random.randn(D)  # randomly initialize the weights
b = 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


P_Y_given_X = forward(X, W, b)
prediction = np.round(P_Y_given_X)

def claissfication_rate(Y, P):
    return np.mean(Y == P)

print ("score: ", claissfication_rate(Y, prediction))