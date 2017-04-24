import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return np.exp(x)/(1+np.exp(x))**2

np.random.seed(0)

nIn = 1
nHidden = 10
nOut = 1
alpha = 0.01
sampleSize = 1;

def train(X, Y, W1, W2):
    #Forward propogation
    print("Input: ", X.shape)
    l2 = np.dot(X, W1)
    print("l2: ", l2.shape)
    l3 = np.dot(sigmoid(l2), W2)
    print("l3: ", l3.shape)

    #Back propogation & loss calculation
    loss3 = sigmoidPrime(l3) * (Y - sigmoid(l3)) #try without sigmoid
    print("loss3: ", loss3.shape)
    loss2 = sigmoidPrime(l2) * np.dot(loss3, W2.transpose())
    print("loss2: ", loss2.shape)
    loss1 = sigmoidPrime(X) * np.dot(loss2, W1.transpose())
    print("loss1: ", loss1.shape)

    #Update weights
    #print("W1: ", np.asmatrix(np.dot(sigmoid(l2), loss2)).shape)
    #W1 = W1 + alpha * np.asmatrix(np.dot(sigmoid(l2), loss1))
    #print("W2: ", np.dot(np.asmatrix(sigmoid(l3)).transpose(), np.asmatrix(loss3)).shape)
    #W2 = W2 + alpha * np.dot(np.asmatrix(sigmoid(l3)).transpose(), np.asmatrix(loss2))

    return W1, W2, np.abs(loss2).sum()

def predict(X, Y, W1, W2):
    #Forward propogation
    l1 = X * W1
    l2 = sigmoid(l1) * W2
    #print(W1.size)
    if Y == l2.sum():
        return 1
    else:
        return 0

W1 = np.random.rand(nIn, nHidden) * 0.5
print("W1 (Original): ", W1.shape)
W2 = np.random.rand(nHidden, nOut) * 0.5
print("W2 (Original): ", W2.shape)

for i in range(sampleSize):
    z = np.random.choice(2, 1).reshape(1, 1)
    W1, W2, loss = train(z, 1-z, W1, W2)
    if i % 1000 == 0:
        print(loss)

count = 0
for i in range(sampleSize):
    z = np.random.choice(2, 1).reshape(1, 1)
    if predict(z, 1-z, W1, W2) == 1:
        #print("Predict: ", predict(z, 1-z, W1, W2))
        count += 1;

print("Accuracy: ", count/100000)
