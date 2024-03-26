# Ejercicio 3: 
# • Winedataset 
# • XGBoost (py-xgboost) 
# • Calcular score CV (3 folds)

import numpy as np
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate


X, y = load_wine(return_X_y=True)
estimator = XGBClassifier(n_estimators=20)
cv_results = cross_validate(estimator, X, y, cv=5)
print(f"Cross validation results: {cv_results}, CV score: {np.mean(cv_results['test_score'])}")
