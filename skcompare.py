from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import pickle
import numpy
import matplotlib.pyplot as plt

def parseData():
    # Load batches
    batch1 = unpickle("data/data_batch_1")
    batch2 = unpickle("data/data_batch_2")
    batch3 = unpickle("data/data_batch_3")
    batch4 = unpickle("data/data_batch_4")
    batch5 = unpickle("data/data_batch_5")

    print(">> Creating the unholy frankenstein")
    X, y = format(batch1, batch2, batch3, batch4, batch5)

    print(">> Summoning and binding test demons")
    # Load test data and format
    test = unpickle("data/test_batch")
    X_test, y_test = format(test)
    return X, y, X_test, y_test

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

# Formatting data into a more digestable form
def format(*batch) -> list:
    X = []
    y = []

    # Casting all values TO and int from uint8
    for b in batch:
        for i in range(len(b[b"data"])):
            X.append([float(x) for x in b[b'data'][i]])
            y.append(float(b[b'labels'][i]))
    return X, y

X, y, X_test, y_test = parseData()
models = []
models.append(LinearRegression())
models.append(LogisticRegression())
models.append(ElasticNet())
y_eval = numpy.array(y_test)
for model in models:
    model.fit(X, y)
    score = model.score(X_test, y_test)
    print(score)
    predict = model.predict(X_test)
    cm = confusion_matrix(y_test, predict)
    ConfusionMatrixDisplay.from_predictions(y_test, predict)
    plt.show()
