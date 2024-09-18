# Parte 1: Predicción del valor de mercado
# a) Recta de regresión para predecir el valor  de mercado de un jugador a partir de la
# característica más relevante (a la que se destinará mayor
# proporción del presupuesto),
# respaldada por:
# i) Prueba de significancia de regresión, coeficiente de determinación (R²)
# y correlación lineal (r).

# ii) Inferencias sobre los parámetros de la recta, estimando las
# fluctuaciones con una confianza del 95%.

# iii) La proporción de veces que el valor de mercado supera la incertidumbre
# de predicción
# comparada con la respuesta media del valor de mercado para una
# característica fija, ambas
# con la misma confianza y ancho mínimo.

from numpy import sqrt
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("code/dataset/wii.csv")

relevant_feature = "NumMinutos"
# Seleccionar las columnas de interés
df = df[[relevant_feature, "Bateria"]].dropna()

# Eliminar todos los valores cuyo overall sea menor a 60
# ya que al ser tan pocos no aportan información relevante
# df = df[df[relevant_feature] >= 65]

# Definir las variables
x = df[relevant_feature]
y = df['Bateria']

# Grafico con los datos obtenidos
# plt.scatter(x, y, color='red')
# plt.title('Value vs Overall')
# plt.xlabel('Overall')
# plt.ylabel('Value')
# plt.show()

# Calcular la sumatoria de las variables x e y
x_sum = x.sum()
y_sum = y.sum()

# Calcular la media de las columnas 2 y 3
x_avg = x.mean()
y_avg = y.mean()

# Calcular la sumatoria de los cuadrados de las variables x e y
x_sq_sum = (x ** 2).sum()
y_sq_sum = (y ** 2).sum()

# Sigma al cuadrado de x e y (varianza corregida)
x_var = x_sq_sum / len(x) - (x_sum ** 2)
y_var = y_sq_sum / len(y) - (y_sum ** 2)

# Sigma de xy
xy_dev = ((x * y).sum() / len(x)) - (x_avg * y_avg)

# Coeficientes de la recta de regresión Pendiente y Ordenada
b1 = xy_dev / x_var
b0 = y_avg - b1 * x_avg

# Estimación de minimos cuadrados
Sxy = (x * y).sum() - ((x.sum() * y.sum()) / len(x))
Sxx = x_sq_sum - ((x.sum() ** 2) / len(x))
Syy = y_sq_sum - ((y.sum() ** 2) / len(y))

# Coeficientes de la recta de regresión Pendiente y Ordenada
b1_hat = Sxy / Sxx
b0_hat = y_avg - b1_hat * x_avg

y_hat = b0_hat + b1_hat * x

# Dibujar recta a partir de y_hat
plt.scatter(x, y, color='red', label='Datos')
plt.plot(x, y_hat, color='blue', label='Recta de regresión')
plt.title('Value vs Overall con Recta de Regresión')
plt.xlabel('Overall')
plt.ylabel('Value')
plt.legend()
plt.show()

SSr = Syy - (b1_hat * Sxy)

# Calculo de la varianza
sigma_hat_sqrd = SSr / (len(x) - 2)

# Calculo del coeficiente de determinación
R2 = 1 - SSr/Syy

# Coeficiente de Correlación Lineal
r = Sxy / sqrt(Sxx * Syy)

# Imprimir los resultados
print("SSr: " + str(SSr))
print("Varianza: " + str(sigma_hat_sqrd))
print("Probabilidad: " + str(R2 * 100) + "%")
print("Coeficiente de correlación: " + str(r) + "\n")
