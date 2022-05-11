from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import pickle
import numpy
import matplotlib.pyplot as plt
import datas

data = datas.getData("hotSplitData")

X_train = data["X_train"]
y_train = datas.getLabelFromHot(data["y_train"], 0)
X_val = data["X_val"]
y_val = datas.getLabelFromHot(data["y_val"], 0)
X_test = data["X_test"]
y_test = datas.getLabelFromHot(data["y_test"], 0)
X = X_train + X_val
y = y_train + y_val

models = []

models.append(LogisticRegression(penalty="none"))
models.append(KNeighborsClassifier(n_neighbors=1))
models.append(BernoulliNB())
models.append(Perceptron())
for model in models:
    model.fit(X, y)
    score = model.score(X_test, y_test)
    print(score)
    predict = model.predict(X_test)
    cm = confusion_matrix(y_test, predict)
    ConfusionMatrixDisplay.from_predictions(y_test, predict)
    plt.savefig("skresults/"+type(model).__name__)
