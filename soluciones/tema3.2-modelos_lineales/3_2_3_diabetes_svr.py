# Ejercicio 3: 
# • Diabetes dataset 
# • Linear Regression 
# • Calcular score test

from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

#from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#estimator = LinearRegression()
estimator = Pipeline([('s', StandardScaler()), ('r', SVR())])
estimator.fit(X_train, y_train)
print(f"{(estimator.score(X_test, y_test))}")

