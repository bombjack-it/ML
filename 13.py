from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# # Delete all NaN row ath the origin
water = pd.read_csv('water_potability.csv', 
                     low_memory=False
                    ).dropna()

X = water.loc[:, water.columns != 'Potability']
y = water["Potability"]

result = []

sgd = linear_model.SGDClassifier()
for epoch in range(0,5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    sgd.fit(X_train,y_train)
    result.append(sgd.score(X_test, y_test))

plt.subplot(1, 1, 1)
plt.plot(result)
plt.show()