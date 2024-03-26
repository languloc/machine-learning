# Ejercicio 2: 
# • Diabetes dataset 
# • Randomsearch MLP 
# • Revisar resultados de la búsqueda

#! /usr/bin/env python

from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform


X, y = load_diabetes(return_X_y=True)

base_estimator = Pipeline([("std", StandardScaler()), ("mlp", MLPRegressor())])
parameters = {'mlp__hidden_layer_sizes': [[10], [20, 10], [30, 20, 10]],
              'mlp__learning_rate_init': uniform(0, 2)}
estimator = RandomizedSearchCV(base_estimator, parameters, n_iter=4)
estimator.fit(X, y)
print(f"{estimator.cv_results_}")
