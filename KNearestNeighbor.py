import numpy
import pickle

class KNearestNeighbor:
    def __init__(self, data: list) -> None:
        self.data = data

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
            c[k[i]] = self._classify(p, k[i], distances)

        # Dictionary of {k: label}
        return c
    
    # Helper function that does the actual calculations
    # Allows for multiple K values to test with
    def _classify(self, p: list, k: int, distances: list) -> float:
        sum = 0
        for i in range(0, k):
            sum += distances[i][0][1]
        sum = sum / k

        return sum

if __name__ == "__main__":
    # Load CIFAR data function
    def unpickle(file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding = "bytes")

        return dict
    
    # Load batch 1
    batch1 = unpickle("data/data_batch_1")
    
    # Formatting data into a more digestable form
    batch1_f = []
    for i in range(0, 10000):
                          # Casting all values TO and int from uint8
        batch1_f.append( [numpy.array([int(x) for x in batch1[b'data'][i]]), batch1[b'labels'][i]] )

    # Load test data and format
    test = unpickle("data/test_batch")

    test_f = []
    for i in range(0, 10000):
                        # Casting all values TO and int from uint8
        test_f.append( [numpy.array([int(x) for x in test[b'data'][i]]), test[b'labels'][i]] )
    
    # Create KNN object
    K3 = KNearestNeighbor(batch1_f)
    
    print(test_f[0][1], end = " | ") # ith label
    print(test_f[0][0]) # list of rgb values [r, r, r, .... g, b, b]

    print(f"test_f[0] classified as {K3.classify(test_f[0][0], [1, 3, 10, 100, 1000])}")