# Ejercicio 1: 
# • Iris dataset 
# • KNN 
# • Calcular score de train, test y CV (3 folds)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier


X, y = load_iris(return_X_y=True)

n_neighbors = 5
estimator = KNeighborsClassifier(n_neighbors=n_neighbors)

estimator.fit(X, y)
score = estimator.score(X, y)
print(f"Train accuracy: {score}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
estimator.fit(X_train, y_train)
score = estimator.score(X_test, y_test)
print(f"Test accuracy: {score}")

cv_results= cross_validate(estimator, X=X, y=y, cv=3)
print(f"Cross validation results: {cv_results}, CV accuracy: {np.mean(cv_results['test_score'])}")
