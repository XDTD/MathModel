import os
import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import  RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random


class PA2:
    def __init__(self, estimator):
        self.data, self.label = self.preprocess_data(estimator)
        self.estimator = estimator

    def preprocess_data(self, estimator):

        names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "label"]

        f = open("adult.csv", "w")

        fdata = open("adult.data", "r")
        ftest = open("adult_test.data")
        k = 0

        for row in fdata:
            row = row.replace(" ", "")
            # print(row)
            f.write(row)
            k += 1
            # if(k == 20):
            #    break
        for row in ftest:
            row = row.replace(" ", "")
            f.write(row)

        # print(k)
        f.close()

        datadf = pd.read_csv("adult.csv", header=None, na_values=['?'], names=names)

        del datadf["workclass"]

        del datadf["race"]

        del datadf["native-country"]

        del datadf["fnlwgt"]

        data = self.makeBinaryIfPosbl(datadf.dropna())
        label = data.pop(">50K")
        del data["<=50K"]

        return data, label

    def makeBinaryIfPosbl(self, dframe):

        # print(dframe)
        binaryListForEachUniqueValue = pd.DataFrame()

        # get type of the columns and if its not float,
        # then we

        for curr in dframe.columns:
            ctype = dframe[curr].dtype
            # print(dframe[curr])
            # print(ctype) object or float
            if ctype != float:

                # print(dframe[curr].value_counts().index, "value")

                # go through each unique value in each of the classes
                # and make true for that value and false for all other values
                # i.e. a special list for each unique value in which if that
                # value is present then true, else false.
                # Apparently thats what I got after searching online
                # Do this and feed to train function to estimate using sklearn

                for c in dframe[curr].value_counts().index:
                    # print(dframe[curr], (dframe[curr] == c))
                    # print(curr, dframe[curr].value_counts().index, c," khatm")
                    # print(dframe[curr], dframe[curr]==c)
                    # print(c," c over \n")
                    binaryListForEachUniqueValue[c] = (dframe[curr] == c)

                    # print(dframe[curr].value_counts().index)
                    # print(curr,"currrrrrr")
            elif ctype == np.int or ctype == np.float:
                binaryListForEachUniqueValue[curr] = dframe[curr]
            else:
                print("unused curr: {}".format(curr))
        # print(binaryListForEachUniqueValue)
        return binaryListForEachUniqueValue

    # Common procedure for algorithms. split, fit, predict
    def train(self, n_examples=None):

        X = self.data.values.astype(np.float32)
        y = self.label.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.estimator.fit(X_train, y_train)

        y_pred = self.estimator.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["<=50k", ">50k"]))

        y_score = self.estimator.predict_proba(X_test)
        print("roc: {}".format(roc_auc_score(y_test, y_score[:, 1])))

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:,1])
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b',
        label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


if __name__ == "__main__":
    seed = np.random.randint(100000)

    #estimator = RandomForestClassifier(n_estimators=5, max_depth=5)
    #estimator = RandomForestClassifier(n_estimators=5, max_depth=50)
    #estimator = RandomForestClassifier(n_estimators=5, max_depth=100)
    #estimator = RandomForestClassifier(n_estimators=50, max_depth=5)
    estimator = RandomForestClassifier(n_estimators=50, max_depth=50)
    #estimator = RandomForestClassifier(n_estimators=50, max_depth=100)
    #estimator = RandomForestClassifier(n_estimators=100, max_depth=5)
    #estimator = RandomForestClassifier(n_estimators=100, max_depth=50)
    #estimator = RandomForestClassifier(n_estimators=100, max_depth=100)

    pa2 = PA2(estimator)

    pa2.train()
