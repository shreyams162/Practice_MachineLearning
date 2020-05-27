import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.info()
train["Outlet_Size"].value_counts()
test.info()

def missing_values(df):
    df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace = True)
    df["Outlet_Size"].fillna("Medium", inplace = True)

missing_values(train)
missing_values(test)

train = pd.get_dummies(train, columns = ["Item_Identifier", "Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"], drop_first = True)
test = pd.get_dummies(test, columns = ["Item_Identifier", "Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"], drop_first = True)

X = train.drop("Item_Outlet_Sales", axis = 1)
Y = train["Item_Outlet_Sales"]
X_test = test    

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred_reg = regressor.predict(X_val)
np.sqrt(mean_squared_error(Y_val, Y_pred_reg))

from sklearn.tree import DecisionTreeRegressor
DT_reg = DecisionTreeRegressor()
DT_reg.fit(X_train, Y_train)
Y_pred_DT = DT_reg.predict(X_val)
np.sqrt(mean_squared_error(Y_val, Y_pred_DT))

from sklearn.ensemble import RandomForestRegressor
RF_reg = RandomForestRegressor(n_estimators = 30)
RF_reg.fit(X_train, Y_train)
Y_pred_RF = RF_reg.predict(X_val)
np.sqrt(mean_squared_error(Y_val, Y_pred_RF))

#Final Submission
Item_Outlet_Sales = RF_reg.predict(X_test)
test = test[["Item_Identifier", "Outlet_Identifier"]]
test1 = pd.read_csv("test.csv")
test["Item_Outlet_Sales"] = Item_Outlet_Sales
test["Item_Identifier"] = test1["Item_Identifier"]
test["Outlet_Identifier"] = test1["Outlet_Identifier"]
test.to_csv("Submission.csv", index = False)