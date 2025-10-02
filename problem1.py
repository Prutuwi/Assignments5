import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
df = data['frame']

plt.show()
sns.heatmap(df.corr().round(2), annot=True)
plt.show()
plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')

plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')

plt.show()

X = df[['bmi', 's5']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print("RMSE = ", rmse)
print("R2 = ", r2)

X2 = df[['bmi', 's5','bp','s5']]
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=50)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_test2 = model.predict(X_test)
y_pred_train2 = model.predict(X_train)
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_new = r2_score(y_test, y_pred_test2)

print("RMSE = ", rmse2)
print("R2 = ", r2_new)

"""
a) The next variable I would add is s6 and bp because they have the highest correlations and adding them both will help to improve the 
performance of the model

b) So far as in my findings the model values haven't changed the way expected. It still stays the same for the RMSE and R2

c) Seems like adding more variable at once doesn't seems to improve the performance of the model anyway it fits the training data better.
"""










