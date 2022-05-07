from typing import *
import numpy
import pickle
import time
import random

class Perceptron:
    def __init__(self, data: List[int], number_of_weights: int, learning_rate: float, iterations: int) -> None:
        self.data = data
        # Initialize weights to random value in [-1, 1]
        self.weights = [random.uniform(-1, 1)] * number_of_weights
        self.n = learning_rate
        self.iterations = iterations
    
    def dot(self, x: List[float]) -> float:
        sum = 0
        for i in range(0, len(x)):
            sum += self.weights[i] * x[i]
        
        return sum
    
    def np_dot(self, x: List[float]) -> float:
        return numpy.dot(self.weights, x)
    
    def sign(self, x: float) -> int:
        if x > 0:
            return 1
        return -1
    
    def computeWeights(self) -> None:
        for j in range(0, self.iterations):
            start_time = time.time()

            # Update each weight for each instance
            for i in range(0, len(self.data)):
                # w_i <- w_i + n(t - o)x
                t = self.data[i][1]
                x = numpy.insert(self.data[i][0], 0, 1)
                o = self.sign(self.np_dot(x))

                # If perceptron's prediction isn't correct update weight
                if o != t:
                    for k in range(0, len(self.weights)):
                        self.weights[k] += self.n * (t - o) * x[k]
            
            print(f">> Weight iteration {j} done in {time.time() - start_time} seconds")
    
    def classify(self, y: List[float]) -> float:
        return self.sign(self.dot(numpy.insert(y, 0, 1)))

if __name__ == "__main__":
    # Load CIFAR data function
    def unpickle(file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding = "bytes")

        return dict
    
    # Formatting data into a more digestable form
    def format(batch: dict) -> list:
        formatted = []
        for i in range(0, 10000):
            # Casting all values TO and int from uint8
            formatted.append( [numpy.array([int(x) for x in batch[b'data'][i]]), batch[b'labels'][i]] )
        
        return formatted
    
    # Load batches
    batch1 = unpickle("data/data_batch_1")
    batch2 = unpickle("data/data_batch_2")
    batch3 = unpickle("data/data_batch_3")
    batch4 = unpickle("data/data_batch_4")
    batch5 = unpickle("data/data_batch_5")

    # Absolute gigabrain ram killing function
    def frankenstein(b1: dict, b2, b3, b4, b5) -> list:
        frankenstein = []
        for i in range(0, 10000):
            # TODO: Cast to int in the euclideanDistance
            frankenstein.append( [numpy.array([int(x) for x in b1[b'data'][i]]), b1[b'labels'][i]] )
            frankenstein.append( [numpy.array([int(x) for x in b2[b'data'][i]]), b2[b'labels'][i]] )
            frankenstein.append( [numpy.array([int(x) for x in b3[b'data'][i]]), b3[b'labels'][i]] )
            frankenstein.append( [numpy.array([int(x) for x in b4[b'data'][i]]), b4[b'labels'][i]] )
            frankenstein.append( [numpy.array([int(x) for x in b5[b'data'][i]]), b5[b'labels'][i]] )
        
        return frankenstein
    
    def birdify(b1: List) -> List:
        birbs = 0
        for i in range(0, len(b1)):
            if b1[i][1] != 2:
                b1[i][1] = -1
            else:
                birbs += 1
                b1[i][1] = 1
    
    start_time = time.time()
    print(">> Loading and formatting data")
    allBatches = frankenstein(batch1, batch2, batch3, batch4, batch5)
    birdify(allBatches)
    print(f">> Data loaded in {time.time() - start_time} seconds")

    # Load test data and format
    test = unpickle("data/test_batch")
    test_f = format(test)
    birdify(test_f)
    
    # Create Preceptron object
    ptron = Perceptron(allBatches, 3072 + 1, 0.1, 3)
    print(">> Computing weights...")
    ptron.computeWeights()

    # Classify some tests
    f = open("results_ptron.csv", "w")
    f.write("instance,lable,predicted,correctness\n")
    correct = 0
    for i in range(0, len(test_f)):
        pred = ptron.classify(test_f[i][0])

        if pred == test_f[i][1]:
            correct += 1

        f.write(f"{i},{test_f[i][1]},{pred},{correct/(i+1)}\n")
        print(f">> Test instance {i}, labeled as {test_f[i][1]}, classified as {pred}, {100 * (correct/(i+1))}% correct")
    
    f.close()