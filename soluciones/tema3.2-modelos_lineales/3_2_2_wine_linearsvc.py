# Ejercicio 2: 
# • Winedataset 
# • Linear SVC 
# • Calcular score test (accuracy, precisión, recall) 
# • Serializar a fichero, deserializar y hacer predicción con test

from joblib import dump, load
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, classification_report


X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

estimator = Pipeline([("std", StandardScaler()),
                      ("lsvc", LinearSVC())])
estimator.fit(X_train, y_train)
acc = estimator.score(X_test, y_test)
print(f"acc: {acc}")

y_pred = estimator.predict(X_test)
print(f"{y_pred}")
prec = precision_score(y_test, y_pred, average="macro")
print(f"prec: {prec}")
rec = recall_score(y_test, y_pred, average="macro")
print(f"rec: {rec}")
report = classification_report(y_test, y_pred)
print(f"{report}")

dump(estimator, 'filename.joblib')
estimator_2 = load('filename.joblib')
y_pred_2 = estimator_2.predict(X_test)
print(f"{y_pred_2}")

