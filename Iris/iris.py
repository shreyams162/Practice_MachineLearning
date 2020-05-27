import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("iris.csv")
X = dataset.iloc[:, 1:5].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test)

#Naive bayes
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)

y_pred_NB = classifier_NB.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_NB = accuracy_score(y_pred, y_test)


