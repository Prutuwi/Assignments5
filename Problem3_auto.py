import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model



df = pd.read_csv("Auto.csv")
print(df)

X = df[['displacement', 'horsepower', 'year', 'weight', 'acceleration', 'cylinders']]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

alphas = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5]

las_scores = []
ridge_scores = []

#Implementation using LASSO and RIDGE regression
for alpha in alphas:
     lasso = linear_model.Lasso(alpha=alpha)
     lasso.fit(X_train, y_train)
     sc = lasso.score(X_test, y_test)
     las_scores.append(sc)


     ridge = Ridge(alpha=alpha)
     ridge.fit(X_train, y_train)
     rc = ridge.score(X_test, y_test)
     ridge_scores.append(rc)


plt.plot(alphas, las_scores)
plt.show()

best_ridge2 = max(ridge_scores)
idx = ridge_scores.index(best_ridge2)
best_rid_alp = alphas[idx]

best_lasso = max(las_scores)
idx = las_scores.index(best_lasso)
best_las_alp = alphas[idx]

print("Best ridge alpha is ", best_rid_alp, " and lasso alpha is ", best_las_alp)
print("Best ridge score is ", best_ridge2, " and lasso score is ", best_lasso)


"""
In the above program as I have completed the ridge and lasso for the Auto.csv
for some several values of alphas. From the outcome found that the best ridge alpha is 5 and
best lasso alpha is 0.5. The best ridge score is 0.757 and the lasso score is 0.763.
"""
