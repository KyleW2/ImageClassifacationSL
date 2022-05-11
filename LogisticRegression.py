from audioop import cross
from copyreg import pickle
from math import log
from scipy.stats import logistic
import numpy as np
import datas
import time
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

#η = eta
#θ = theta
class LogisticModel:
    def __init__(self, N) -> None:
        self.N = N
        self.weights = None
    
    def train(self, X, y, iters):
        self.weights = GradientDescent(X, y, self.N, iters, self.weights)

    def getPredictions(self, X, y):
        predictions = []
        for i in range(len(y)):
            predictions.append(predict(h(X[i], self.weights)))
        return predictions

    def getLoss(self, X, y):
        loss = 0
        for i in range(len(y)):
            loss += crossEntropyLoss(X[i], y[i], self.weights)
        return loss
    
    def getConfusionMatrix(self, X, y):
        h = self.getPredictions(X, y)
        return datas.confusionMatrix(y, h)

def predict(y):
    if y >= .5: return 1
    else: return 0

def h(x, w):
    return sigmoid(np.transpose(w)@x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def crossEntropyLoss(X, y, w):
    p = h(X,w)
    if p == 1: p = .9999999999999999
    if p == 0: p = .0000000000000001
    return -(y * log(p) + (1 - y) * log(1 - p))

def GradientDescent(X,y,N,iters, theta = None):
    num_weights = len(X[0])
    m = len(X)
    if theta == None: theta = [0] * num_weights
    lastLoss = None
    for iter in range(iters):
        loss = 0
        for i in range(m):
            y_hat = h(X[i], theta)
            loss += crossEntropyLoss(X[i], y[i], theta)
            g = [0] * num_weights
            for j in range(num_weights):
                g[j] = (y_hat - y[i]) * X[i][j]
            for w in range(num_weights):
                theta[w] = theta[w] - N * g[w]
        #Check if gradient is changing loss meaningfully
        if lastLoss != None:
            delta_loss = abs(loss - lastLoss)
            if delta_loss < 1:
                print("Loss delta =",delta_loss,"after",iter,"iterations")
                return theta
        lastLoss = loss
        print("Iteration =", iter, "Loss = ", loss)
    return theta

def trainBest():
    data = datas.getData("hotSplitData")

    X_train = data["X_train"]
    y_train = datas.getLabelFromHot(data["y_train"], 0)
    X_val = data["X_val"]
    y_val = datas.getLabelFromHot(data["y_val"], 0)
    X_test = data["X_test"]
    y_test = datas.getLabelFromHot(data["y_test"], 0)
    X = X_train + X_val
    y = y_train + y_val

    models = []

    #models.append(LogisticModel(.000000001))
    models = datas.unpickle("results/LogisticRegression/10iter1E-9model")
    

    for model in models:
        model.train(X, y, 200)
        loss = model.getLoss(X_test, y_test)
        cm = model.getConfusionMatrix(X_test, y_test)
        print("Loss =", loss)
        datas.printConfusionMatrix(cm)

    datas.pickle_data(models, "results/LogisticRegression/10iter1E-9model")

def plotConfusionMatrix():
    data = datas.getData("hotSplitData")
    X_test = data["X_test"]
    y_test = datas.getLabelFromHot(data["y_test"], 0)
    model = datas.unpickle("results/LogisticRegression/10iter1E-9model")
    predictions = model[0].getPredictions(X_test, y_test)
    #cm = model.getConfusionMatrix(predictions, y_test)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.show()

if __name__ == "__main__":
    pass



