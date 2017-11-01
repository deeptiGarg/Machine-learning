# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

challengeReg = linear_model.LinearRegression()
dataframe = pd.read_csv('challenge_dataset.csv')
x_values = dataframe[['X']]
y_values = dataframe[['Y']]

challengeReg.fit(x_values,y_values)

#challengeReg.predict(5.7077)


plt.scatter(x_values, y_values)
plt.plot(5.7077, challengeReg.predict(5.7077))
plt.show()

 

