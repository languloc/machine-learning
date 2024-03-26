# Ejercicio 4: 
# • MNIST dataset 
# • MLP KerasSklearn wrapper 
# • Calcular score de CV

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score


data = load_breast_cancer()
X = data['data']
y = data['target']
n_features = 30
dim_target = 2


def create_nn():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_features,)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(dim_target, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
    return model


classifier = KerasClassifier(build_fn=create_nn, epochs=100)
print(f"CV score: {(cross_val_score(classifier, X, y).mean())}")
