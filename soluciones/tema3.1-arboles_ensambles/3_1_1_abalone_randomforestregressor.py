# Ejercicio 1: 
# • Abalone dataset (OpenML) 
# • RandomForest 
# • Calcular score CV (3 folds)

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor


X, y = fetch_openml(name="abalone", return_X_y=True)
# La variable "Sex" puede tomar valores M (Male), F (Female) o I (Infant), y
# parece que lo mejor es que I tome un valor intermedio entre M y F puesto que
# puede acabar en cualquiera de esas dos clases pasado un tiempo
X["Sex"] = X["Sex"].apply(lambda x: {"M": 1.0, "I": 0.0, "F": -1.0}[x])
y = y.astype(np.float32)
estimator = RandomForestRegressor(n_estimators=20, max_depth=4)
#estimator = DecisionTreeRegressor(max_depth=2)

cv_results = cross_validate(estimator, X=X, y=y, cv=3)
print(f"Cross validation results: {cv_results}, CV score: {np.mean(cv_results['test_score'])}")
