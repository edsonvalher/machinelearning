# documentación: https://scikit-learn.org/stable/datasets/index.html
# regresión lineal simple
"""
    Este ejercicio se trata de calcular el precio de las casas de boston de acuerdo
    al número de habitaciones que cuenta la vivienda
"""
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# importamos los datos del dataset
casasboston = datasets.load_boston()
print(casasboston)
print()

print("estructura del dataset")
print(casasboston.keys())
print()

# descripci´n de datos
print(casasboston.DESCR)

# cantidad de datos
print(casasboston.data.shape)


# nombre de las columnas
# RM será la columna que nos da el número de habitaciones
print(casasboston.feature_names)

# predictor de regresion lineal simple
# las columna 5 contiene las variables
# los datos son almacenados en un PIE debe tratarse como tal
X = casasboston.data[:, np.newaxis, 5]
y = casasboston.target


# el modelo de regresion lineal por los puntos tendrá un error muy alto

# 1 separar los datos en entrenamiento y prueba
# utilizaremos un 20% de los datos como prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# definimos el algoritmo a utilizar lineal regresion

lr = linear_model.LinearRegression()
# realizamos el entrenamiento
lr.fit(X_train, y_train)
# realizo una predicción
y_pred = lr.predict(X_test)

plt.scatter(X, y)
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.title('Regresión lineal simple')
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')

print()
print('DATOS DEL MODELO DE REGRESION LINEAL SIMPLE')
print()
print('El valor de la pendiente o coeficiente "a":')
print(lr.coef_)
print('El valor de la intersección o coeficiente "b":')
print(lr.intercept_)
print()
print('La ecuación del modelo es igual a:')
print('y=', lr.coef_, ' x ', lr.intercept_)
print()
# calcula pa precisión del algoritmo
print("Precisión del algoritmo")
# el resultado dió 0.47 que es igual a un 47% dandonos una idea que no es el algoritmo ideal para este calculo
print(lr.score(X_train, y_train))

plt.show()

# calculamos los valores de la pendiente
