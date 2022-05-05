from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.linear_model import LogisticRegression
import pickle

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
            X.append([int(x) for x in b[b'data'][i]])
            y.append(b[b'labels'][i])
    return X, y

X, y, X_test, y_test = parseData()
model1 = LinearRegression()
model2 = LogisticRegression()
model3 = ElasticNet()

for model in model1, model2, model3:
    model.fit(X, y)
    score = model.score(X_test, y_test)
    predict = model.predict(X_test)
    print(score)

while True: x = 0