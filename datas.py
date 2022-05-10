import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np

fullyFormattedFileName = "formattedData"
#3-way-split
#Data = 50,000
#Train = 40,000
#Validation = 10,000
#Test = 10,000
def parseData(samples_per_batch = None, formatted = False, save = False):
    # Load batches
    batch1 = unpickle("data/data_batch_1")
    batch2 = unpickle("data/data_batch_2")
    batch3 = unpickle("data/data_batch_3")
    batch4 = unpickle("data/data_batch_4")
    batch5 = unpickle("data/data_batch_5")

    print(">> Creating the unholy frankenstein")
    X, y = format(batch1, batch2, batch3, batch4, batch5, samples_per_batch=samples_per_batch)


    print(">> Summoning and binding test demons")
    # Load test data and format
    test = unpickle("data/test_batch")
    X_test, y_test = format(test)
    data = {"X" : X, "y" : y, "X_test" : X_test, "y_test" : y_test}
    return data

def pickle_data(obj, filename):
    import pickle
    file =  open("data/" + filename, "wb")
    pickle.dump(obj, file)

def unpickle(file):
    import pickle
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

def oneHot(label):
    y = [0] * 10
    y[label] = 1
    return y

def hotOnes(data):
    Y = []
    for y in data:
        Y.append(oneHot(y))
    return Y

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def hotNormalDataInYourArea(data):
    new_data = {"X" : NormalizeData(data["X"]), "y" : hotOnes(data["y"]), "X_test" : NormalizeData(data["X_test"]), "y_test" : hotOnes(data["y_test"])}
    pickle_data(new_data, fullyFormattedFileName)
    return new_data

# Formatting data into a more digestable form
def format(*batch, samples_per_batch = None) -> list:
    X = []
    y = []

    # Casting all values TO and int from uint8
    for b in batch:
        for i in range(len(b[b"data"]) if samples_per_batch == None else samples_per_batch):
            X.append([int(x) for x in b[b'data'][i]])
            y.append(int(b[b'labels'][i]))
    return X, y

def vectorToImage(X):
    img = np.reshape(X, (3,32,32))
    imgView = np.transpose(img, (1,2,0))
    plt.imshow(imgView)
    plt.show()

def getData():
    try: return unpickle("data/" + fullyFormattedFileName)
    except: return parseData()

if __name__ == "__main__":
    data = getData()
    pass

def printConfusionMatrix(cm):
    print("True Negative:",cm[0][0])
    print("False Positive:",cm[0][1])
    print("False Negative:",cm[1][0])
    print("True Positive:", cm[1][1])

def confusionMatrix(y, h):
    true_neg = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    for i in range(len(y)):
        if y[i] == h[i]:
            if h[i] == 0: true_neg += 1
            else: true_pos += 1
        else:
            if h[i] == 0: false_neg += 1
            else: false_pos += 1
    return [[true_neg, false_pos], [false_neg, true_pos]]