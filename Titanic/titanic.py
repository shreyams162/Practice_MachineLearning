import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()
train.info()
train["Age"].describe()
train["Cabin"].describe()

train["Age"].fillna(30, inplace = True)
train["Age"].isnull().sum()
train["Age"] = train["Age"].astype(int)
train["Ticket"].describe()

test.info()
test["Age"].describe()
test["Age"].fillna(30, inplace = True)
test["Age"] = test["Age"].astype(int)
test["Fare"].fillna(0, inplace = True)

PassengerId = test["PassengerId"]
train = train.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis = 1)
test = test.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis = 1)

train = pd.get_dummies(train, columns = ["Sex", "Embarked"], drop_first = True)
test = pd.get_dummies(test, columns = ["Sex", "Embarked"], drop_first = True)

X = train.drop("Survived", axis = 1).values
Y = train["Survived"].values

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.20, random_state = 0, shuffle = True)

from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, Y_train)
Y_pred_LR = classifier_LR.predict(X_val)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_val, Y_pred_LR)
acc_LR = (cm[0][0] + cm[1][1]) / 179

from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, Y_train)
Y_pred_NB = classifier_NB.predict(X_val)
cm_NB = confusion_matrix(Y_val, Y_pred_NB)
acc_NB = (cm_NB[0][0] + cm_NB[1][1]) / 179

from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier()
classifier_DT.fit(X_train, Y_train)
Y_pred_DT = classifier_DT.predict(X_val)
cm_DT = confusion_matrix(Y_val, Y_pred_DT)
acc_DT = (cm_DT[0][0] + cm_DT[1][1]) / 179

from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 150)
classifier_RF.fit(X_train, Y_train)
Y_pred_RF = classifier_RF.predict(X_val)
cm_RF = confusion_matrix(Y_val, Y_pred_RF)
acc_RF = (cm_RF[0][0] + cm_RF[1][1]) / 179

from sklearn.svm import SVC
classifier_SVM = SVC(kernel = "linear", random_state = 0)
classifier_SVM.fit(X_train, Y_train)
Y_pred_SVM = classifier_SVM.predict(X_val)
cm_SVM = confusion_matrix(Y_val, Y_pred_SVM)
acc_SVM = (cm_SVM[0][0] + cm_SVM[1][1]) / 179

Survived = classifier_RF.predict(test)
test1 = pd.read_csv("test.csv")
test["PassengerId"] = test1["PassengerId"]
test["Survived"] = Survived
test.to_csv("Submission.csv", index = False)

