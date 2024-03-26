# Ejercicio 3: 
# • Breast cancer dataset 
# • RFCV con regresión logística 
# • Encontrar número óptimo de features

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)

lr = LogisticRegression(max_iter=5000)
rfecv = RFECV(estimator=lr, step=1, cv=StratifiedKFold(2),
              scoring='recall')

rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)

print(rfecv.__dict__)
print(rfecv.estimator.__dict__)