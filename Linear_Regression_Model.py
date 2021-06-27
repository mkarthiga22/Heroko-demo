import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn import linear_model
le = preprocessing.LabelEncoder()
df=pd.read_csv('D:/machine learning/insurance.csv')
print(df)
le.fit(df["sex"])
df["Sex"] = le.transform(df["sex"])
le.fit(df["smoker"])
df["Smoker"] = le.transform(df["smoker"])
le.fit(df["region"])
df["Region"] = le.transform(df["region"])
print(df)
y = df["charges"]
x = df[["age", "bmi", "children", "Sex", "Smoker", "Region"]]
print(y)
print(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)

predictions = linreg.predict(X_test)
print(predictions)
linreg.score(X_test,y_test)
print(linreg.score)