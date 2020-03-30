import  numpy as np
import  pandas as pd

import os.path

data_folder = os.path.dirname('/home/talg/udemy_deep_learning/machine_learning_examples/ann_logistic_extra/')
data_file = 'ecommerce_data.csv'
data_path = os.path.join(data_folder, data_file)

def get_data():
    df = pd.read_csv(data_path)

    # print (df.head())
    # numpy array
    data = df.values

    # shuffle it
    np.random.shuffle(data)

    # split features and lables
    X = data[:, :-1]    # all table except last column
    Y = data[:, -1].astype(np.int32)     # all rows, last column (user action)

    # print ("X: ", X)
    # print ("Y: ", Y)

    # one-hot encode the categorical value
    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    # one hot
    for n in range(N):
        t = int(X[n, D-1])  # time of day
        X2[n, t+D-1] = 1

    # one hot different way
    # Z = np.zeros(N, 4)
    # Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    # X2[:, -4] = Z

    # split traing and test
    Xtrain = X2[:-100]
    Ytrain = Y[:-100]

    Xtest = X2[-100:]
    Ytest = Y[-100:]

    # normalize columns 1 and 2
    for i in (1, 2):
        m = Xtrain[: i].mean()
        s = Xtrain[: i].std()
        Xtrain[:, i] = (Xtrain[:, i] - m) / s
        Xtest[:, i] = (Xtest[:, i] - m) / s

    return Xtrain, Ytrain, Xtest, Ytest

def get_binary_data():
    # return only the data from the first 2 classes
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    X2train = Xtrain[Ytrain <= 1]
    Y2train = Ytrain[Ytrain <= 1]
    X2test = Xtest[Ytest <= 1]
    Y2test = Ytest[Ytest <= 1]

    return X2train, Y2train, X2test, Y2test


# print (get_data())



