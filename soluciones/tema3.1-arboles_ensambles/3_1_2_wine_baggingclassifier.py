# Ejercicio 2: 
# • Winedataset 
# • Bagging of MLPs 
# • Calcular score CV (3 folds)

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X, y = load_wine(return_X_y=True)
estimator = BaggingClassifier(estimator=Pipeline([("std", StandardScaler()),
                                                  ("mlp", MLPClassifier(hidden_layer_sizes=[40, 20],
                                                                        max_iter=1000))]),
                              n_estimators=20)
cv_results = cross_validate(estimator, X, y, cv=5)
print(f"Cross validation results: {cv_results}, CV score: {np.mean(cv_results['test_score'])}")
