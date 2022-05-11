from math import log
import datas
import time

class NaiveBayesModel:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.m = len(X)

    def predict(self, X, labels):
        max = None
        for label in labels:
            p_y = self.prob(label)
            ynb = log(p_y)
            for i in range(len(X)):
                p_yx = self.prob(label, X[i], i)
                ynb+=log(p_yx)
            #print("Ynb=",ynb,"Label=",label,"Prob y",p_y,"Prob x given y",p_yx)
            if max == None or ynb > max:
                max = ynb
                best_y = label
        return best_y

    def m_estimate(nc, n, p, m):
        return (nc + m * p) / (n + m)

    def prob(self, y, x= None, index = None):
        num_y = 0
        ##Probability of y
        if x == None:
            for i in range(self.m):
                if self.y[i] == y: num_y += 1
            return num_y/self.m
        ##Probability of x given y
        num_x = 0
        for i in range(self.m):
            if self.y[i] == y:
                num_y += 1
                if self.X[i][index] == x: num_x += 1
        m_estimate = NaiveBayesModel.m_estimate(num_x, num_y, 1/256, self.m)
        return m_estimate

if __name__ == "__main__":
    data = datas.getData("hotSplitData")
    X_train = data["X_train"]
    y_train = datas.getLabelFromHot(data["y_train"], 0)
    X_val = data["X_val"]
    y_val = datas.getLabelFromHot(data["y_val"], 0)
    X_test = data["X_test"]
    y_test = datas.getLabelFromHot(data["y_test"], 0)
    X = X_train + X_val
    y = y_train + y_val

    model = NaiveBayesModel(X, y)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    correct_predictions = 0
    for i in range(len(y_test)):
        predictions = []
        #correct = "Incorrect"
        #starttime = time.time()
        prediction = model.predict(X_test[i], (0, 1))
        predictions.append(prediction)
        #if prediction == y_test[i]: 
            #correct = "Incorrect"
            #if y_test[i] == 1: TP += 1
            #else: TN += 1
        #else:
            #if y_test[i] == 1: FP += 1
            #else: FN += 1

        #print("Prediction",i, "took", time.time()-starttime,"seconds. Prediction:", prediction, "True value:",y_test[i])
        
    #print("TP:",TP,"FP:",FP,"TN:",TN,"FN:",FN)
        if i % 50 == 0:
            datas.pickle_data(predictions, "BayesPredictions")

    datas.pickle_data(predictions, "BayesPredictions")