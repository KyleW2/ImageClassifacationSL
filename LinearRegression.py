from math import log
import numpy
import pickle
import time

class Instance:
    def __init__(self, attributes, label=None) -> None:
        self.attributes = attributes
        self.label = label
        
class LinearRegressionModel:
    def __init__(self, n_features, n_labels) -> None:
        self.n_features = n_features
        self.n_labels = n_labels
        #Initialize Weights
        self.weights = [0] * (n_features + 1)

    def train(self, trainingExamples, n, iterations):
        print("Training on", len(trainingExamples), "examples")
        self.weights = self.GradientDecent(trainingExamples, n, iterations)

    def weightOutput(self, w, x):
        sum = 0
        for i in range(len(x)):
            sum += w[i] * x[i]
        return sum
    
    def predict(self, example):
        prediction = self.weightOutput(self.weights, [1] + list(example.attributes))
        #print("Predicition is", prediction)
        return prediction

    def getPredictionAccuracy(self, examples):
        print("Predicting accuracy of", len(examples), "examples")
        correct = 0
        
        for example in examples:
            #print("Correct label is", example.label)
            if self.predict(example) - example.label: correct += 1
        return correct/len(examples)
            

    def GradientDecent(self, trainingExamples, n, iterations):
        dim = len(trainingExamples[0].attributes) + 1
        w = self.weights
        for iter in range(iterations):
            dW = [0] * dim
            for example in trainingExamples:
                x = [1] + list(example.attributes)
                t = example.label
                
                for i in range(dim):
                    o = self.weightOutput(w, x)
                    dW[i] += n * (t - o) * x[i]
            for j in range(dim):
                w[j] += dW[j]
        return w

    def empRisk(self, sample):
        risk = 0
        for example in sample:
            h = self.weightOutput(self.weights, example.attributes)
            c = example.label
            risk += self.sqLoss(h,c)
        return risk / len(sample)

    def sqLoss(self, h, c):
        return (h - c)**2

    def logLoss(self, h, x, y, p):
        loss = 0
        for i in self.n_labels:
            loss += y[i] * log(p[i])
        return -loss

    # Find the distance between two vectors
    def eulideanDistance(self, l1: list, l2: list) -> float:
        sum = 0
        for i in range(0, len(l1)):
            sum += (l2[i] - l1[i]) ** 2

        return sum ** (1/2)
    
    # Return a list of disances in ascending order
    def calculateDistances(self, p: list) -> list:
        distances = []

        for i in range(0, len(self.data)):
            # Append (array of rgb, disance between) to list
            distances.append( (self.data[i], self.eulideanDistance(self.data[i][0], p)) )
        
        distances.sort(key = lambda x: x[1])
        return distances
    
    # Classify a vector
    def classify(self, p: list, k: list) -> dict:
        distances = self.calculateDistances(p)
        
        c = {}
        for i in range(0, len(k)):
            c[k[i]] = self._classify(k[i], distances)

        # Dictionary of {k: label}
        return c
    
    # Helper function that does the actual calculations
    # Allows for multiple K values to test with
    def _classify(self, k: int, distances: list) -> float:
        sum = 0
        for i in range(0, k):
            sum += distances[i][0][1]
        sum = sum / k
        return sum

# Formatting data into a more digestable form
def format(*batch, n) -> list:
    formatted = []
    for i in range(0, n):
        # Casting all values TO and int from uint8
        for b in batch:
            formatted.append( Instance([int(x) for x in b[b'data'][i]], b[b'labels'][i]) )
    return formatted

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

def parseData():
    # Load batches
    batch1 = unpickle("data/data_batch_1")
    batch2 = unpickle("data/data_batch_2")
    batch3 = unpickle("data/data_batch_3")
    batch4 = unpickle("data/data_batch_4")
    batch5 = unpickle("data/data_batch_5")

    print(">> Creating the unholy frankenstein")
    allBatches = format(batch1, batch2, batch3, batch4, batch5, n=10000)

    print(">> Summoning and binding test demons")
    # Load test data and format
    test = unpickle("data/test_batch")
    test_f = format(test, n = 10000)
    return allBatches, test_f

if __name__ == "__main__":
    # Load CIFAR data function
    print("Loading Data...")
    trainingData, testData = parseData() 
    print(">> Forging Linear Model")
    model = LinearRegressionModel(3072, 10)
    accuracy = 0
    risk = 100
    while risk > 1:
        model.train(trainingData, .000000001, 1)
        #accuracy = model.getPredictionAccuracy(testData)
        risk = model.empRisk(testData)
        print("Empirical Risk is", model.empRisk(testData))

