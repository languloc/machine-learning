# Ejercicio 1: 
# • Diabetes dataset 
# • MLP 
# • Calcular score (R2, MAE, MSE) CV (3 folds)

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)

regressor = Pipeline([("std", StandardScaler()),
                      ("mlp", MLPRegressor(hidden_layer_sizes=(200, 20, 10),
                                           max_iter=1000))])
scores = cross_validate(regressor, X, y, cv=3, scoring=("r2",
                                                        "neg_mean_absolute_error",
                                                        "neg_mean_squared_error"))
print(f"{scores}")
print(f"R2: {np.mean(scores['test_r2'])}")
print(f"MAE: {-np.mean(scores['test_neg_mean_absolute_error'])}")
print(f"MSE: {-np.mean(scores['test_neg_mean_squared_error'])}")