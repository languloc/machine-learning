# Ejercicio 1: 
# • Abalone dataset 
# • Grid search Random Forest 
# • Revisar resultados de la búsqueda

#! /usr/bin/env python

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


X, y = fetch_openml(name="abalone", return_X_y=True)
# La variable "Sex" puede tomar valores M (Male), F (Female) o I (Infant), y
# parece que lo mejor es que I tome un valor intermedio entre M y F puesto que
# puede acabar en cualquiera de esas dos clases pasado un tiempo
X["Sex"] = X["Sex"].apply(lambda x: {"M": 1.0, "I": 0.0, "F": -1.0}[x])
parameters = {'n_estimators': [10, 20], 'max_depth': [2, 4]}
estimator = GridSearchCV(RandomForestRegressor(), parameters)
estimator.fit(X, y)
print(f"{estimator.cv_results_}")
