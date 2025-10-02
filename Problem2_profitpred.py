import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


import seaborn as sns



from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


df = pd.read_csv('50_Startups.csv')

print(df)

plt.scatter(df['R&D Spend'], df['Profit'])

sns.heatmap(data=df.corr(numeric_only=True), annot=True)

plt.show()


plt.scatter(df['R&D Spend'], df['Profit'])

plt.show()

X = pd.DataFrame(df[['R&D Spend']], columns=['R&D Spend'])
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Train MSE:", rmse_train)
print("Test MSE:", rmse_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train R^2:", r2_train)
print("Test R^2:", r2_test)

"""
In the above I have choosen R&D Spend and Profit as the variables for investigating the correlation because they show 
a more positive correlation and as a result I was able to find that trained RMSE is 9788 and test RMSE is 6557.
The trained R2 is 0.93 and test R2 is 0.97

"""