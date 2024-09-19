import pandas as pd
import numpy as np

# * b) Ecuación para predecir el valor de mercado del jugador
# * a partir de varias características.

# Cargar el dataset
df = pd.read_csv("code/dataset/FIFA21.csv")

# Seleccionar las columnas de interés
y_key = 'value_eur'
x1_key = 'overall'
x2_key = 'potential'
df = df[[y_key, x1_key, x2_key]].dropna()

# Filtrar datos con overall mayor a 65
df = df[df["overall"] > 65]
# df = df[df["potential"] > 65]

# * i) Usando el método de mínimos cuadrados. Explica los indicadores obtenidos
# *  (como el coeficiente de determinación y la correlación) y proporciona una
# * breve interpretación de los resultados


# Definir las variables
x1 = df[x1_key].values
x2 = df[x2_key].values
y = df[y_key].values

x1_sum = x1.sum()
x2_sum = x2.sum()
y_sum = y.sum()

print(y.mean())

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
print(f"Coeficiente b1 (Overall): {b1}")
print(f"Coeficiente b2 (Age): {b2}")

# Predicción usando la ecuación de regresión múltiple
y_hat = b0 + b1 * x1 + b2 * x2

# Imprimir predicciones
print(f"Predicciones: {y_hat[:5]}")  # Mostrar solo las primeras 5 predicciones
# Imprimir la ecuación de regresión múltiple
print(f"Ecuación de regresión múltiple: y_hat = {b0} + {b1}*x1 + {b2}*x2")


# Calcular R^2
ss_res = np.sum((y - y_hat) ** 2)  # Suma de los residuos al cuadrado
ss_tot = np.sum((y - y.mean()) ** 2)  # Suma total de los cuadrados
r2 = 1 - (ss_res / ss_tot)

print(f"Coeficiente de determinación (R²): {r2}")

corr_overall = np.corrcoef(x1, y)[0, 1]
corr_potential = np.corrcoef(x2, y)[0, 1]

print(f"Correlación entre overall y value_eur: {corr_overall}")
print(f"Correlación entre potential y value_eur: {corr_potential}")


# * ii) Usando el método de descenso por gradiente. ¿Son los valores obtenidos iguales a los
# * conseguidos mediante la resolución del sistema de ecuaciones normales? Muestra los
# * resultados obtenidos junto con las últimas iteraciones del algoritmo. Indica los valores de los
# * parámetros utilizados (como tasa de aprendizaje y número de iteraciones)


# * iii) Da una interpretación del criterio de corte utilizado en el algoritmo del gradiente. Explica
# * si presenta alguna falla. Si no es una buena condición de corte, ¿puedes sugerir un criterio
# * alternativo más eficaz?
