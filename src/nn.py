import numpy as np

np.random.seed(0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))

data = np.load("src/irisData.npz")
irisX = data["irisX"]
irisY = data["irisY"]

nIn = 4
nHidden = 5
nOut = 1
alpha = 1.0
sampleSize = 500;

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

    return W1, W2, np.sum((Y-al3)**2)/nIn * 100

def predict(X, W1, W2):
    #Forward propogation
    l2 = np.dot(X, W1)
    l3 = np.dot(sigmoid(l2), W2)
    return sigmoid(l3)

######################################################## Function Calls Start

W1 = np.random.rand(nIn, nHidden) * 0.5
W2 = np.random.rand(nHidden, nOut) * 0.5

for j in range(sampleSize):
    for i in range(irisX.shape[0] - 15):
        W1, W2, loss = train(irisX[i].reshape(1, 4), irisY[i].reshape(1, 1)/2, W1, W2)
    if j % 50 == 0:
        print("Loss: ", loss)

for i in range(15):
    print("P: ", predict(irisX[i+135], W1, W2)*2, " -> ", round(np.asscalar(predict(irisX[i+135], W1, W2)*2)), " : ", np.asscalar(irisY[i+135]))
