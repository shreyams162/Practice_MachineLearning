import numpy as np
import pandas as pd

dataset = pd.read_csv("AB_NYC_2019.csv")

dataset.info()
dataset["reviews_per_month"].describe()
dataset["reviews_per_month"].fillna(dataset["reviews_per_month"].mean(), inplace = True)
dataset["reviews_per_month"].isnull().sum()

dataset = pd.get_dummies(dataset, columns = ["neighbourhood_group", "neighbourhood", "room_type"], drop_first = True)

X = dataset.drop(["id", "name", "host_id", "host_name", "latitude", "longitude", "price", "last_review"], axis = 1).values
Y = dataset["price"].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0, shuffle = True)

from sklearn.linear_model import LinearRegression
reg_LR = LinearRegression()
reg_LR.fit(X_train, Y_train)
Y_pred_LR = reg_LR.predict(X_test)

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(Y_test, Y_pred_LR))

from sklearn.tree import DecisionTreeRegressor
reg_DT = DecisionTreeRegressor().fit(X_train, Y_train)
Y_pred_DT = reg_DT.predict(X_test)
np.sqrt(mean_squared_error(Y_test, Y_pred_DT))

from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor(n_estimators = 50).fit(X_train, Y_train)
Y_pred_RF = reg_RF.predict(X_test)
np.sqrt(mean_squared_error(Y_test, Y_pred_RF))
