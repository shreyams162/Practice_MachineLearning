import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.info()
train["education"].value_counts()
train["previous_year_rating"].value_counts()
test.info()
train["previous_year_rating"].describe()

train["previous_year_rating"].fillna(0, inplace = True)
train["previous_year_rating"] = train["previous_year_rating"].astype(int)
train["education"].fillna("Bachelor's", inplace = True)
test["previous_year_rating"].fillna(0, inplace = True)
test["previous_year_rating"] = test["previous_year_rating"].astype(int)
test["education"].fillna("Bachelor's", inplace = True)

train.isnull().sum()
test.isnull().sum()

train = pd.get_dummies(train, columns = ["department", "region", "education", "gender", "recruitment_channel"], drop_first = True)
test = pd.get_dummies(test, columns = ["department", "region", "education", "gender", "recruitment_channel"], drop_first = True)

X = train.drop(["is_promoted", "employee_id"], axis = 1)
Y = train["is_promoted"]
X_test = test.drop("employee_id", axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.25, random_state = 0, shuffle = True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
classifier_LR = LogisticRegression(random_state = 0)
classifier_LR.fit(X_train, Y_train)
Y_pred_LR = classifier_LR.predict(X_val)
f1_score(Y_val, Y_pred_LR)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_val, Y_pred_LR)

from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
classifier_KNN.fit(X_train, Y_train) 
Y_pred_KNN = classifier_KNN.predict(X_val)
f1_score(Y_val, Y_pred_KNN)

from sklearn.svm import SVC
classifier_SVM = SVC(kernel = "linear", random_state = 0)
classifier_SVM.fit(X_train, Y_train)
Y_pred_SVM = classifier_SVM.predict(X_val)
f1_score(Y_val, Y_pred_SVM)

from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, Y_train)
Y_pred_NB = classifier_NB.predict(X_val)
f1_score(Y_val, Y_pred_NB)

from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, Y_train)
Y_pred_RF = classifier_RF.predict(X_val)
f1_score(Y_val, Y_pred_RF)

from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DT.fit(X_train, Y_train)
Y_pred_DT = classifier_DT.predict(X_val)
f1_score(Y_val, Y_pred_DT)

is_promoted = classifier_RF.predict(X_test)
test1 = pd.read_csv("test.csv")
test["employee_id"] = test1["employee_id"]
test["is_promoted"] = is_promoted
test.to_csv("Submission.csv", index = False)