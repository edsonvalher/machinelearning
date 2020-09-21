# predecir el precio de las casas en boston de acuerdo el numero de habitaciones,
# tiempo ocupado y distancia
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

casasboston = datasets.load_boston()
# preparacion de data lineal multiple
X_multiple = casasboston.data[:, 5:8]
# print(X_multiple)
y_multiple = casasboston.target

# separamos en entrenamiento y prueba
# utilizaremos un 20% de los datos como prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_multiple, y_multiple, test_size=0.2)


lr_multiple = linear_model.LinearRegression()
# realizamos el entrenamiento
lr_multiple.fit(X_train, y_train)
# realiza la predicción
Y_pred_multiple = lr_multiple.predict(X_test)
# print()
#print("--------- data original -------")
# print(y_test)
#print("--------- data multiple -------")
# print(Y_pred_multiple)
print('El valor de la pendiente o coeficiente "a":')
print(lr_multiple.coef_)
print('El valor de la intersección o coeficiente "b":')
print(lr_multiple.intercept_)
print("Precisión del algoritmo")
# el resultado dió 0.5349 que es igual a un 53% dandonos una idea que no es el algoritmo ideal para este calculo
# tiene que estar lo mas cercano a 1 para ser el indicado
print(lr_multiple.score(X_train, y_train))
