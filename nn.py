import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

np.random.seed(0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))

iris = datasets.load_iris()
irisX = np.asarray(iris.data[:, :]).reshape(150, 4)  # we only take the first two features.
irisY = np.asarray(iris.target).reshape(150, 1)

permutation = np.random.permutation(irisX.shape[0])

irisX = irisX[permutation]
irisY = irisY[permutation]

nIn = 4
nHidden = 5
nOut = 1
alpha = 0.1
sampleSize = 2000;

def train(X, Y, W1, W2):
    #Forward propogation
    l2 = np.dot(X, W1)
    al2 = sigmoid(l2)
    l3 = np.dot(al2, W2)
    al3 = sigmoid(l3)

    #Update weights
    delta3 = -(Y - al3) * sigmoidPrime(l3)
    deltaW2 = np.dot(al2.T, delta3)

    delta2 = np.dot(delta3, W2.T) * sigmoidPrime(l2)
    deltaW1 = np.dot(X.T, delta2)

    W2 = W2 - alpha * deltaW2
    W1 = W1 - alpha * deltaW1
    #W2 = W2 - alpha * al3 * (al3 - Y) * al3 * (1 - al3)
    #W1 = W1 - alpha * al2 * np.dot((al3 - Y) * al3 * (1 - al3), W2.transpose()) * al2 * (1 - al2)

    return W1, W2, loss

def predict(X, W1, W2):
    #Forward propogation
    l2 = np.dot(X, W1)
    l3 = np.dot(sigmoid(l2), W2)
    return sigmoid(l3)

W1 = np.random.rand(nIn, nHidden) * 0.5
W2 = np.random.rand(nHidden, nOut) * 0.5
for j in range(sampleSize):
    for i in range(irisX.shape[0] - 15):
        W1, W2, loss = train(irisX[i].reshape(1, 4), irisY[i].reshape(1, 1)/2, W1, W2)
    if j % 200 == 0:
        print("Loss: ", loss)

for i in range(15):
    print("P: ", predict(irisX[i+135], W1, W2)*2, " -> ", round(np.asscalar(predict(irisX[i+135], W1, W2)*2)))
    print("A: ", irisY[i+135])
