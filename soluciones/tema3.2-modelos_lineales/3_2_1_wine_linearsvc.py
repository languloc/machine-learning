# Ejercicio 1: 
# • Winedataset 
# • Linear SVC 
# • Calcular score (accuracy, f1macro) CV (3 folds)

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


X, y = load_wine(return_X_y=True)

estimator = Pipeline([("std", StandardScaler()), ("lsvc", LinearSVC())])
cv_results = cross_validate(estimator, X, y, scoring=["accuracy", "f1_macro"], cv=3)
print(cv_results)
print(
    f"Cross validation results: {cv_results}, CV accuracy: {np.mean(cv_results['test_accuracy'])}, CV f1_macro: {np.mean(cv_results['test_f1_macro'])}"
)
