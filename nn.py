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

def train(X, Y, W1, W2):
    #Forward propogation
    l1 = X * W1
    l2 = sigmoid(l1) * W2

    #Back propogation & loss calculation
    loss2 = sigmoidPrime(l2) * (Y - sigmoid(l2))**2
    loss1 = sigmoidPrime(l1) * (loss2 * W2)**2
    #print("Loss: ", np.sum(loss2))

    #Update weights
    W1 = W1 + alpha * (X * loss1)
    W2 = W2 + alpha * (l1 * loss2)

    return W1, W2, np.abs(loss2).sum()

def predict(X, Y, W1, W2):
    #Forward propogation
    l1 = X * W1
    l2 = sigmoid(l1) * W2
    print(l2.size)
    if Y == l2.sum():
        return 1
    else:
        return 0

W1 = np.random.normal(scale=0.1, size=(nIn, nHidden))
W2 = np.random.normal(scale=0.1, size=(nHidden, nOut))

for i in range(10000):
    z = np.random.choice(2, 1)
    W1, W2, loss = train(z, 1-z, W1, W2)
    if i % 1000 == 0:
        print(loss)

count = 0
for i in range(10000):
    z = np.random.choice(2, 1)
    if predict(z, 1-z, W1, W2) == 1:
        print("Predict: ", predict(z, 1-z, W1, W2))
        count += 1;

print("Accuracy: ", count/100000)
