import numpy
import pickle

class KNearestNeighbor:
    def __init__(self, data: list, k: int) -> None:
        self.data = data
        self.k = k

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
    def classify(self, p: list) -> int:
        distances = self.calculateDistances(p)

        sum = 0
        for i in range(0, self.k):
            sum += distances[i][0][1]
        sum = sum / self.k

        return sum

if __name__ == "__main__":
    # Load CIFAR data function
    def unpickle(file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding = "bytes")

        return dict
    
    # Load batch 1
    batch1 = unpickle("data/data_batch_1")

    """ 
    # Printing the first 10 for reading
    for i in range(0, 10):
        print(batch1[b'labels'][i], end = " | ") # ith label
        print(batch1[b'data'][i]) # list of rgb values [r, r, r, .... g, b, b]
    """
    
    # Formatting data into a more digestable form
    batch1_f = []
    for i in range(0, 10000):
                          # Casting all values TO and int from uint8
        batch1_f.append( [numpy.array([int(x) for x in batch1[b'data'][i]]), batch1[b'labels'][i]] )

    # Create KNN object where k = 3
    K3 = KNearestNeighbor(batch1_f, 10)

    # Load test data and format
    test = unpickle("data/test_batch")

    test_f = []
    for i in range(0, 10000):
                        # Casting all values TO and int from uint8
        test_f.append( [numpy.array([int(x) for x in test[b'data'][i]]), test[b'labels'][i]] )
    
    print(test_f[0][1], end = " | ") # ith label
    print(test_f[0][0]) # list of rgb values [r, r, r, .... g, b, b]

    print(f"test_f[0] classified as {K3.classify(test_f[0][0])}")