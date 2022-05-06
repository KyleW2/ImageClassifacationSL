import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np

def parseData(samples_per_batch = None):
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
    return X, y, X_test, y_test

def unpickle(file):
    import pickle
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

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

X, y, X_test, y_test = parseData(100)
y_plt = []
# for i in range(10):
#     y_plt.append([])
#     for x in X[i]:
#         y_plt[i].append(y[i])

def vectorToImage(X):
    img = np.reshape(X, (3,32,32))
    imgView = np.transpose(img, (1,2,0))
    plt.imshow(imgView)
    plt.show()
# for i in range(len(X)):
#     plt.scatter(sum(X[i]),y[i], 1)

# plt.show()

def compress(X):
    pca_dims = PCA()
    pca_dims.fit(X)
    cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1