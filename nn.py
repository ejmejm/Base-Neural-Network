import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return np.exp(x)/(1+np.exp(x))**2

np.random.seed(0)

nIn = 2
nHidden = 10
nOut = 2
alpha = 0.01
sampleSize = 100;

def train(X, Y, W1, W2):
    #Forward propogation
    l2 = np.dot(X, W1)
    l3 = np.dot(sigmoid(l2), W2)
    print(l3)
    print("A: ", Y)

    #Back propogation & loss calculation
    loss3 = sigmoidPrime(l3) * (Y - sigmoid(l3))
    loss2 = sigmoidPrime(l2) * np.dot(loss3, W2.transpose())
    loss1 = sigmoidPrime(X) * np.dot(loss2, W1.transpose())

    #Update weights
    W1 = W1 + alpha * np.dot(X.transpose(), np.ones((X.transpose().shape[1], W1.shape[1])))
    W2 = W2 + alpha * np.dot(l2.transpose(), np.ones((l2.transpose().shape[1], W2.shape[1])))

    return W1, W2, np.average(np.abs(loss2))

def predict(X, W1, W2):
    #Forward propogation
    l2 = np.dot(X, W1)
    l3 = np.dot(sigmoid(l2), W2)
    return l3

W1 = np.random.rand(nIn, nHidden) * 0.5
W2 = np.random.rand(nHidden, nOut) * 0.5

for i in range(sampleSize):
    z = np.random.choice(2, nIn).reshape(1, nIn)
    W1, W2, loss = train(z, 1-z, W1, W2)
    #if i % 10000 == 0:
    #    print(loss)

z = np.random.choice(2, nIn).reshape(1, nIn)
print("Input: ", z)
print("Output: ", predict(z, W1, W2))

#count = 0
#for i in range(sampleSize):
    #z = np.random.choice(2, nIn).reshape(1, nIn)

    #if predict(z, 1-z, W1, W2) == 1:
    #    print("Predict: ", predict(z, 1-z, W1, W2))
    #    count += 1;

#print("Accuracy: ", count/sampleSize)
