# Ejercicio 3: 
# • Diabetes dataset 
# • Randomsearch MLP 
# • Revisar resultados de la búsqueda y evaluar (nested CV) 
# • Serializar y deserializar el modelo

import joblib
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV


X, y = load_diabetes(return_X_y=True)

_predictor = Pipeline([('std', StandardScaler()),
                       ('bagging', BaggingRegressor(base_estimator=MLPRegressor(max_iter=20)))])

predictor = GridSearchCV(_predictor,
                         {'bagging__base_estimator__hidden_layer_sizes': [(20, 10),
                                                                          (30, 20, 10)],
                          'bagging__base_estimator__alpha': [0.001, 0.01],
                          'bagging__base_estimator__solver': ['adam', 'sgd']})

cv_score = cross_val_score(predictor, X, y, cv=KFold(n_splits=5, shuffle=True))
print(str(cv_score))

predictor.fit(X, y)
joblib.dump(predictor, 'predictor.joblib')
