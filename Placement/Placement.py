import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("Placement_Data_Full_Class.csv")

dataset.info()
dataset.describe()

dataset.fillna(0, inplace = True)
dataset.isnull().sum()

corr = dataset.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

dataset = pd.get_dummies(dataset, columns = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"], drop_first = True)

X = dataset.drop(["sl_no", "status"], axis = 1)
Y = dataset["status"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0, shuffle = True)

from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, Y_train)
Y_pred_LR = classifier_LR.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(Y_test, Y_pred_LR)
acc_LR = (cm_LR[0][0] + cm_LR[1][1]) / 54