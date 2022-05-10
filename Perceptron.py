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
            
            #print(f">> Weight iteration {j+1} done in {time.time() - start_time} seconds")
    
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
    
    def binaryify(b1: List, label: int) -> List:
        new_data = []

        for i in range(0, len(b1)):
            if b1[i][1] == label:
                new_data.append([b1[i][0], 1])
            else:
                new_data.append([b1[i][0], -1])
        
        return new_data

    def runTests(ptron: Perceptron, test_data: list, label: int, number_of_tests: int, number_of_weight_updates: int) -> None:
        for j in range(0, number_of_tests):
            start_time = time.time()
            ptron.computeWeights(number_of_weight_updates)
            ptron.saveWeights(f"weights/ptron_label_{label}_weights_iterated_{(j+1) * number_of_weight_updates}_times")
            # Classify some tests
            f = open(f"results/ptron_label_{label}_weights_iterated_{(j+1) * number_of_weight_updates}_times.csv", "w")
            f.write("instance,lable,predicted,tp,tn,fp,fn,correctness\n")
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i in range(0, len(test_data)):
                pred = ptron.classify(test_data[i][0])

                if test_data[i][1] == 1 and pred == 1:
                    tp += 1
                elif test_data[i][1] == -1 and pred == -1:
                    tn += 1
                elif test_data[i][1] == -1 and pred == 1:
                    fp += 1
                elif test_data[i][1] == 1 and pred == -1:
                    fn += 1


                f.write(f"{i},{test_data[i][1]},{pred},{tp},{tn},{fp},{fn},{(tp + tn)/(i+1)}\n")
            
            print(f">> Test {j+1} of perceptron with label {label} done in {time.time() - start_time} seconds")
            
            f.close()

    start_time = time.time()
    print(">> Loading data...")
    # Load batches
    batch1 = unpickle("data/data_batch_1")
    batch2 = unpickle("data/data_batch_2")
    batch3 = unpickle("data/data_batch_3")
    batch4 = unpickle("data/data_batch_4")
    batch5 = unpickle("data/data_batch_5")
    allBatches = frankenstein(batch1, batch2, batch3, batch4, batch5)

    # Load test data
    test = unpickle("data/test_batch")
    test = format(test)

    print(f">> Done in {time.time() - start_time} seconds.")

    print(">> Formatting data...")
    start_time = time.time()

    # Formatting training data
    airplanes = binaryify(allBatches, 0)
    automobiles = binaryify(allBatches, 1)
    birds = binaryify(allBatches, 2)
    cats = binaryify(allBatches, 3)
    deer = binaryify(allBatches, 4)
    dogs = binaryify(allBatches, 5)
    frogs = binaryify(allBatches, 6)
    horses = binaryify(allBatches, 7)
    ships = binaryify(allBatches, 8)
    trucks = binaryify(allBatches, 9)

    # Formatting test data
    airplanes_test = binaryify(test, 0)
    automobiles_test = binaryify(test, 1)
    birds_test = binaryify(test, 2)
    cats_test = binaryify(test, 3)
    deer_test = binaryify(test, 4)
    dogs_test = binaryify(test, 5)
    frogs_test = binaryify(test, 6)
    horses_test = binaryify(test, 7)
    ships_test = binaryify(test, 8)
    trucks_test = binaryify(test, 9)

    print(f">> Done in {time.time() - start_time} seconds.")

    # Create Preceptron objects
    print(">> Creating perceptrons...")
    start_time = time.time()

    airplanes_ptron = Perceptron(airplanes, 3072 + 1, 0.1)
    automobiles_ptron = Perceptron(airplanes, 3072 + 1, 0.1)
    birds_ptron = Perceptron(birds, 3072 + 1, 0.1)
    cats_ptron = Perceptron(cats, 3072 + 1, 0.1)
    deer_ptron = Perceptron(deer, 3072 + 1, 0.1)
    dogs_ptron = Perceptron(dogs, 3072 + 1, 0.1)
    frogs_ptron = Perceptron(frogs, 3072 + 1, 0.1)
    horses_ptron = Perceptron(horses, 3072 + 1, 0.1)
    ships_ptron = Perceptron(ships, 3072 + 1, 0.1)
    trucks_ptron = Perceptron(trucks, 3072 + 1, 0.1)

    print(f">> Done in {time.time() - start_time} seconds.")

    # Running tests
    print(">> Running tests...")
    start_time = time.time()

    runTests(airplanes_ptron, airplanes_test, 0, 10, 1)
    runTests(automobiles_ptron, automobiles_test, 1, 10, 1)
    runTests(birds_ptron, birds_test, 2, 10, 1)
    runTests(cats_ptron, cats_test, 3, 10, 1)
    runTests(deer_ptron, deer_test, 4, 10, 1)
    runTests(dogs_ptron, dogs_test, 5, 10, 1)
    runTests(frogs_ptron, frogs_test, 6, 10, 1)
    runTests(horses_ptron, horses_test, 7, 10, 1)
    runTests(ships_ptron, ships_test, 8, 10, 1)
    runTests(trucks_ptron, trucks_test, 9, 10, 1)

    print(f">> Done in {time.time() - start_time} seconds.")