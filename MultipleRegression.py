import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("C:/Users/Anu/Desktop/Advertising.csv.txt")
data.drop(['Unnamed : 0'], axis=1)

plt.figure(figsize=(16,8))
plt.scatter(
    data['TV'],
    data['sales'],
    c = 'black'
    )
plt.xlabel("Money on TV ads ($)")
plt.ylabel("Sales($)")
plt.show()
X = data.drop(['sales', 'Unnamed : 0'], axis=1)
Y = data['sales'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X,Y)
print("The linear model is: Y = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))


    
