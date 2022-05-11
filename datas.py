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
    return new_data

def hotData(data):
    new_data = {"X" : data["X"], "y" : hotOnes(data["y"]), "X_test" :data["X_test"], "y_test" : hotOnes(data["y_test"])}
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

def getLabelFromHot(y, label_index):
    new_y = []
    for i in range(len(y)):
        new_y.append(y[i][label_index])
    return new_y

def getData(filename):
    try: return unpickle("data/" + filename)
    except: return None

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

def splitData(data):     
    X = data["X"]
    y = data["y"]
    X_train = X[:30000]
    y_train = y[:30000]
    X_val = X[30000:]
    y_val = y[30000:]
    X_test = data["X_test"]
    y_test = data["y_test"]
    new_data = {"X_train" : X_train, "y_train" : y_train, "X_val" : X_val, "y_val" : y_val, "X_test" : X_test, "y_test" : y_test}
    return new_data

def pickleAllData():
    data = parseData()
    pickle_data(data, "parsedData")
    hot_data = hotData(data)
    pickle_data(hot_data, "hotData")
    hot_split_data = splitData(hot_data)
    pickle_data(hot_split_data, "hotSplitData")

def confusionResults(TP, FP, TN, FN):
    recall = TP/ (TP + FN)
    precision = TP / (TP + FP)
    specificity = TN / (TN + FP)
    youden = recall - (1 - specificity)
    print("Recall:",recall)
    print("Precision:",precision)
    print("Specificity:",specificity)
    print("Youden:",youden)
if __name__ == "__main__":
    TP = 140
    FP = 860
    TN = 8878
    FN = 122
    confusionResults(TP,FP,TN,FN)

