

import joblib
from sklearn.datasets import load_diabetes


X, y = load_diabetes(return_X_y=True)

predictor = joblib.load('predictor.joblib')
print(str(predictor.predict(X)))
