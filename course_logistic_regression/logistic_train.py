import numpy as np
import matplotlib.pyplot as plt

# from sklearn.utils import suffle
from process import get_binary_data

X, Y, _, _ = get_binary_data()

Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
W = np.random.randn(D)
b = 0


def sigmoid(z):
    1 / (1 + np.exp(-z))


def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


def classification_rate(Y, P):
    return np.mean(Y == P)


def cross_entropy(T, pY):
    return -np.mean(T * np.log(pY) + (1 - T) * np.log(1 - pY))

train_cost = []
test_cost = []
learning_rate = 0.001

for i in range(1000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    # calculate the training cost
    ctrain = cross_entropy(Ytrain, pYtrain)
    # calculate the test cost
    ctest = cross_entropy(Ytest, pYtest)

    train_cost.append(ctrain)
    test_cost.append(ctest)

    # now we do gradiant decent
    W -= learning_rate*Xtrain.T.dot(pyTrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()

    if i % 1000:
        print ("i: ", i, ", ctrain: ", ctrain, ", ctest: ", ctest)

print ("final train classification rate: ", classification_rate(Ytrain, np.round(pYtrain)))
print ("final test classification rate: ", classification_rate(Ytest, np.round(pYtest)))

legend1 = plt.plot(train_cost, label="train cost")
legend2 = plt.plot(test_cost, label="test cost")

plt.legend([legend1, legend2])
plt.show()