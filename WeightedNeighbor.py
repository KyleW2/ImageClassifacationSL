import numpy
import pickle
import time

class WeightedNeighbor:
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
            c[k[i]] = self._classify(k[i], distances)

        # Dictionary of {k: label}
        return c
    
    # Helper function that does the actual calculations
    # Allows for multiple K values to test with
    def _classify(self, k: int, distances: list) -> float:
        numerator = 0
        denominator = 0
        for i in range(0, k):
            w = 1 / (distances[i][1] ** 2)
            # w * c(x)
            numerator += w * distances[i][0][1]
            denominator += w

        return numerator / denominator

if __name__ == "__main__":
    # Load CIFAR data function
    def unpickle(file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding = "bytes")

        return dict
    
    # Load batches
    batch1 = unpickle("data/data_batch_1")
    batch2 = unpickle("data/data_batch_2")
    batch3 = unpickle("data/data_batch_3")
    batch4 = unpickle("data/data_batch_4")
    batch5 = unpickle("data/data_batch_5")
    
    # Formatting data into a more digestable form
    def format(batch: dict) -> list:
        formatted = []
        for i in range(0, 10000):
            # Casting all values TO and int from uint8
            formatted.append( [numpy.array([int(x) for x in batch[b'data'][i]]), batch[b'labels'][i]] )
        
        return formatted

    """
    batch1_f = format(batch1)
    batch2_f = format(batch2)
    batch3_f = format(batch3)
    batch4_f = format(batch4)
    batch5_f = format(batch5)
    """

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
    
    print(">> Creating the unholy frankenstein")
    allBatches = frankenstein(batch1, batch2, batch3, batch4, batch5)

    print(">> Summoning and binding test demons")
    # Load test data and format
    test = unpickle("data/test_batch")
    test_f = format(test)
    
    print(">> Forging KNN object")
    # Create KNN object
    K3 = WeightedNeighbor(allBatches)
    
    """
    print(test_f[0][1], end = " | ") # ith label
    print(test_f[0][0]) # list of rgb values [r, r, r, .... g, b, b]
    """

    print(">> Classifying tests")
    
    f = open("results_weighted.csv", "w")
    f.write("test_instance,label,1,3,5,10,50,100,500,1000\n")
    for i in range(0, 10000):
        start_time = time.time()
        results = K3.classify(test_f[i][0], [1, 3, 5, 10, 50, 100, 500, 1000])
        f.write(f"{i},{test_f[i][1]},{results[1]},{results[3]},{results[5]},{results[10]},{results[50]},{results[100]},{results[500]},{results[1000]}\n")
        print(f">> {i}th instance classified in {time.time() - start_time} seconds")
    f.close()