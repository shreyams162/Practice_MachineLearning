#Admission SVM

import numpy as np
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Admission_Predict_ver1.1.csv")
X = dataset.iloc[:, 1:8]
y = dataset.iloc[:, 8]

#Converting floating point values into 0 and 1 for logistic regression
for i in range(0,500):
    if(y[i] < 0.5):
        y[i] = 0
    else:
        y[i] = 1

#Spliting data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Using label encoder to convert float to int for test and train data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelen = LabelEncoder()
y_train_encoded = labelencoder.fit_transform(y_train)
y_test_encoded = labelen.fit_transform(y_test)
      
#Fitting logistic regression to data
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train_encoded)

#Prediction
y_pred = classifier.predict(X_test)

#Using Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_encoded, y_pred)

#Gausian kernel svm gives 95% accuracy

