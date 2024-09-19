import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv("code/dataset/FIFA21.csv")

# Seleccionar las columnas de interés
y_key = 'value_eur'
x1_key = 'overall'
x2_key = 'potential'
df = df[[y_key, x1_key, x2_key]].dropna()

# Filtrar datos con overall mayor a 65
# df = df[df["overall"] > 65]

# Definir las variables
x1 = df[x1_key].values
x2 = df[x2_key].values
y = df[y_key].values

x1_sum = x1.sum()
x2_sum = x2.sum()
y_sum = y.sum()

# Calcular la suma de los cuadrados de x1 y x2
x1_square = (x1 ** 2).sum()
x2_square = (x2 ** 2).sum()

# Calcular la suma de los productos de x1 y x2
x1_x2 = (x1 * x2).sum()

# Calcular la suma de los productos de x1 e y
x1_y = (x1 * y).sum()

# Calcular la suma de los productos de x2 e y
x2_y = (x2 * y).sum()
# Resolver el sistema de ecuaciones
A = np.array([
    [len(y), x1_sum, x2_sum],
    [x1_sum, x1_square, x1_x2],
    [x2_sum, x1_x2, x2_square]
])

B = np.array([y_sum, x1_y, x2_y])

beta = np.linalg.solve(A, B)

# Extraer los coeficientes de regresión
b0 = beta[0]  # Intersección
b1 = beta[1]  # Coeficiente de x1
b2 = beta[2]  # Coeficiente de x2

# Mostrar los coeficientes de la regresión
print(f"Coeficiente b0 (intersección): {b0}")
print(f"Coeficiente b1 (PesoIX1): {b1}")
print(f"Coeficiente b2 (PesoAX2): {b2}")

# Predicción usando la ecuación de regresión múltiple
y_hat = b0 + b1 * x1 + b2 * x2

# Imprimir predicciones
print(f"Predicciones: {y_hat[:5]}")  # Mostrar solo las primeras 5 predicciones
# Imprimir la ecuación de regresión múltiple
print(f"Ecuación de regresión múltiple: y_hat = {b0} + {b1}*x1 + {b2}*x2")
