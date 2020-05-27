import numpy as np
import pandas as pd

dataset = pd.read_csv("weight-height.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([("encoder", OneHotEncoder(drop = "first"), [0])], remainder = "passthrough")
X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

X_train = np.asarray(X_train, dtype='float64')
X_test = np.asarray(X_test, dtype='float64')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

