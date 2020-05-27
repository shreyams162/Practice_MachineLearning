import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
meal_info = pd.read_csv("meal_info.csv")
center_info = pd.read_csv("fulfilment_center_info.csv")

train.info()
train.head()

meal_info.info()
center_info.info()

train_V1 = pd.merge(train, meal_info, on = "meal_id", how = "outer")
train_V1 = pd.merge(train_V1, center_info, on="center_id", how = "outer")

test_V1 = pd.merge(test, meal_info, on = "meal_id", how = "outer")
test_V1 = pd.merge(test_V1, center_info, on = "center_id", how = "outer")

train_V1.info()

train_V1 = pd.get_dummies(train_V1, columns = ["category", "cuisine", "center_type"], drop_first = True)
test_V1 = pd.get_dummies(test_V1, columns = ["category", "cuisine", "center_type"], drop_first = True)

X = train_V1.drop(["id", "center_id", "meal_id"], axis = 1)
Y = train_V1["num_orders"]

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.25, random_state = 0, shuffle =True)

from sklearn.linear_model import LinearRegression
reg_LR = LinearRegression()
reg_LR.fit(X_train, Y_train)
Y_pred_LR = reg_LR.predict(X_val)

from sklearn.metrics import mean_squared_log_error
100 * np.sqrt(mean_squared_log_error(Y_val, Y_pred_LR))

from sklearn.tree import DecisionTreeRegressor
reg_DT = DecisionTreeRegressor()
reg_DT.fit(X_train, Y_train)
Y_pred_DT = reg_DT.predict(X_val)
100 * np.sqrt(mean_squared_log_error(Y_val, Y_pred_DT))

from sklearn.ensemble import RandomForestRegressor
reg_RF = RandomForestRegressor(n_estimators = 100)
reg_RF.fit(X_train, Y_train)
Y_pred_RF = reg_RF.predict(X_val)
100 * np.sqrt(mean_squared_log_error(Y_val, Y_pred_RF))

from sklearn.svm import SVR
reg_SVR = SVR(kernel = "linear", C=100, gamma = "auto")
reg_SVR.fit(X_train, Y_train)
Y_pred_SVR = reg_SVR.predict(X_val)
100 * np.sqrt(mean_squared_log_error(Y_val, Y_pred_SVR))

