from audioop import cross
from math import log
from scipy.stats import logistic
import numpy as np
import datas
import time

#η = eta
#θ = theta
class LogisticModel:
    def __init__(self) -> None:
        pass

    def predict(y):
        if y >= .5: return 1
        else: return 0

def h(x, w):
    return sigmoid(np.transpose(w)@x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#How close the classifier output h is to the correct output y
#Conditional maximum liklihood estimation, choose weights and bias that maximize the log probability of the true y labels in the training data given oberservation x
#Neative log likelihood loss, or cross entropy loss
def crossEntropyLoss(X, y, w):
    p = h(X,w)
    if p == 1: p = .9999999999
    if p == 0: p = .0000000001
    return -(y * log(p) + (1 - y) * log(1 - p))

#Gradient Descent
#Find optimal weights by minimizing the loss function we've defined for the model
#Find set of weights which minimizes the loss function, averaged over all examples
#The gradient of a function of many variables is a vector pointing in the direction of the greatest increase of the function
#∇ = gradient
#gradient of a single weight w_j = [σ(w·x+b)−y]x_j
#Online function
#function STOCHASTIC GRADIENT DESCENT(L(), f(), x, y) returns θ
# where: L is the loss function
# f is a function parameterized by θ
# x is the set of training inputs x(1), x(2),..., x(m)
# y is the set of training outputs (labels) y(1), y(2),..., y(m)
#θ ←0
#repeat til done # see caption
#For each training tuple (x(i), y(i)) (in random order)
#1. Optional (for reporting): # How are we doing on this tuple?
#Compute ˆy(i) = f(x(i);θ) # What is our estimated output ˆy?
#Compute the loss L(yˆ(i), y(i)) # How far off is ˆy(i) from the true output y(i)?
#2. g←∇θ L(f(x(i);θ), y(i)) # How should we move θ to maximize loss?
#3. θ ←θ − η g # Go the other way instead
#return θ

def stochasticGradientDescent(X,y,N,iters):
    num_weights = len(X[0])
    m = len(X)
    theta = [0] * num_weights
    for i in range(iters):
        for i in range(m):
            y_hat = h(X[i], theta)
            #loss = crossEntropyLoss(X[i], y[i], theta)
            #print("Loss = ", loss)
            g = [0] * num_weights
            for j in range(num_weights):
                g[j] = (y_hat - y[i]) * X[i][j]
            for w in range(num_weights):
                theta[w] = theta[w] - N * g[w]
    return theta

if __name__ == "__main__":
    data = datas.parseData()
    X = data["X"]
    y = data["y"]
    N = .01
    for i in range(len(y)):
        if i == 0: y[i] = 1
        else: y[i] = 0
    weights = stochasticGradientDescent(X, y, N, 1)
    print(weights)
