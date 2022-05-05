from math import log
import numpy
import pickle
import time

class LogisticModel:
    def __init__(self, n_features, n_labels) -> None:
        self.n_features = n_features
        self.n_labels = n_labels
        #Initialize Weights
        self.weights = [0] * n_features

    def loss(self, h, x, y, p):
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
def format(batch: dict) -> list:
    formatted = []
    for i in range(0, 10000):
        # Casting all values TO and int from uint8
        formatted.append( [numpy.array([int(x) for x in batch[b'data'][i]]), batch[b'labels'][i]] )
    return formatted

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

# Absolute gigabrain ram killing function
def frankenstein(b1: dict, b2, b3, b4, b5) -> list:
    frankenstein = []
    for i in range(0, 10000):
        frankenstein.append( [numpy.array([int(x) for x in b1[b'data'][i]]), b1[b'labels'][i]] )
        frankenstein.append( [numpy.array([int(x) for x in b2[b'data'][i]]), b2[b'labels'][i]] )
        frankenstein.append( [numpy.array([int(x) for x in b3[b'data'][i]]), b3[b'labels'][i]] )
        frankenstein.append( [numpy.array([int(x) for x in b4[b'data'][i]]), b4[b'labels'][i]] )
        frankenstein.append( [numpy.array([int(x) for x in b5[b'data'][i]]), b5[b'labels'][i]] )
    return frankenstein
    
def parseData():
    # Load batches
    batch1 = unpickle("data/data_batch_1")
    batch2 = unpickle("data/data_batch_2")
    batch3 = unpickle("data/data_batch_3")
    batch4 = unpickle("data/data_batch_4")
    batch5 = unpickle("data/data_batch_5")
    """
    batch1_f = format(batch1)
    batch2_f = format(batch2)
    batch3_f = format(batch3)
    batch4_f = format(batch4)
    batch5_f = format(batch5)
    """
    print(">> Creating the unholy frankenstein")
    allBatches = frankenstein(batch1, batch2, batch3, batch4, batch5)

    print(">> Summoning and binding test demons")
    # Load test data and format
    test = unpickle("data/test_batch")
    test_f = format(test)
    return allBatches, test_f

if __name__ == "__main__":
    # Load CIFAR data function
    print("Loading Data...")
    trainingData, testData = parseData() 
    print(">> Forging Logistic Model")
    logisticModel = LogisticModel()
