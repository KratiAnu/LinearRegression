import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("C:/Users/Anu/Desktop/Advertising.csv.txt")
data.drop(['Unnamed : 0'], axis = 1)
plt.figure(figsize=(16,8))
plt.scatter(
    data['TV'],
    data['sales'],
    c = 'black'
    )
plt.xlabel("Money on TV ads ($)")
plt.ylabel("Sales($)")
plt.show()
X = data['TV'].values.reshape(-1,1)
Y = data['sales'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X,Y)
print("The linear model is: Y = {:.5}+{:.5}X".format(reg.intercept_[0],reg.coef_[0][0]))
predictions = reg.predict(X)
plt.figure(figsize = (16,8))
plt.scatter(
    data['TV'],
    data['sales'],
    c = 'black'
    )
plt.plot(
    data['TV'],
    predictions,
    c='blue',
    linewidth=4
    )
plt.xlabel("Money on TV ads ($)")
plt.ylabel("Sales($)")
plt.show()
    
