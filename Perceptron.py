from typing import *
import numpy
import pickle
import time
import random

class Perceptron:
    def __init__(self, data: List[int], number_of_weights: int, learning_rate: float) -> None:
        self.data = data
        # Initialize weights to random value in [-1, 1]
        self.weights = [random.uniform(-1, 1)] * number_of_weights
        self.n = learning_rate
    
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
    
    def computeWeights(self, iterations: int) -> None:
        for j in range(0, iterations):
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
            
            print(f">> Weight iteration done in {time.time() - start_time} seconds")
    
    def classify(self, y: List[float]) -> float:
        return self.sign(self.dot(numpy.insert(y, 0, 1)))
    
    def saveWeights(self, f: str) -> None:
        f = open(f, "w")

        for i in range(0, len(self.weights)):
            f.write(f"{self.weights[i]}\n")
        
        f.close()

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
    ptron = Perceptron(allBatches, 3072 + 1, 0.1)

    def runTests(number_of_tests: int) -> None:
        for j in range(0, number_of_tests):
            start_time = time.time()
            ptron.computeWeights(1)
            ptron.saveWeights(f"ptron_weights_{j+1}_iter")
            # Classify some tests
            f = open(f"results/results_ptron_{j+1}_iter.csv", "w")
            f.write("instance,lable,predicted,tp,tn,fp,fn,correctness\n")
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i in range(0, len(test_f)):
                pred = ptron.classify(test_f[i][0])

                if test_f[i][1] == 1 and pred == 1:
                    tp += 1
                elif test_f[i][1] == -1 and pred == -1:
                    tn += 1
                elif test_f[i][1] == -1 and pred == 1:
                    fp += 1
                elif test_f[i][1] == 1 and pred == -1:
                    fn += 1


                f.write(f"{i},{test_f[i][1]},{pred},{tp},{tn},{fp},{fn},{(tp + tn)/(i+1)}\n")
            
            print(f">> Iteration {j+1} done in {time.time() - start_time} seconds") # Test instance {i}, labeled as {test_f[i][1]}, classified as {pred}, {100 * ((tp + tn)/(i+1))}% correct")
            
            f.close()
    
    print(">> Running tests...")
    runTests(100)