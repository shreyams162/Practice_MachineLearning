import numpy as np
import pandas as pd

train = pd.read_csv("train_ctrUa4K.csv")
test = pd.read_csv("test_lAUu6dG.csv")

X_train = train.iloc[:, 1:12].values
y_train = train.iloc[:, 12].values
X_test = test.iloc[:, 1:12].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder()
X_train = onehotencoder.fit_transform(X_train).toarray()
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 1:12])
X_train[:, 1:12] = imputer.transform(X_train[:, 1:12])

from sklearn_pandas import CategoricalImputer
categorical_imputer = CategoricalImputer(missing_values = 'NaN', strategy = 'most_frequent')
categorical_imputer = categorical_imputer.fit(X_train[:, ])