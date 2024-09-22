# Parte 1: Predicción del valor de mercado

from numpy import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Cargar el dataset
df = pd.read_csv("code/dataset/FIFA21.csv")

relevant_feature = "overall"
# Seleccionar las columnas de interés
df = df[["long_name", "value_eur", relevant_feature]].dropna()


# Eliminar todos los valores cuyo overall sea menor a 60 o mayor a 80 ya que no son relevantes
df = df[df["overall"] > 55]
df = df[df["overall"] < 80]

# Normalizar la columna 'value_eur' a un rango de 0 a 100
df['value_eur_normalized'] = 100 * (df['value_eur'] - df['value_eur'].min()) / (df['value_eur'].max() - df['value_eur'].min())

# Definir las variables
x = df[relevant_feature]
y = df['value_eur_normalized']

# Grafico con los datos obtenidos
# plt.scatter(x, y, color='red')
# plt.title('Value vs Overall')
# plt.xlabel('Overall')
# plt.ylabel('Value')
# plt.show()

# Recta de regresión para predecir el valor de mercado
# de un jugador a partir de la característica más relevante

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
x_var = x_sq_sum / len(x) - (x_sum ** 2) / len(x)**2
y_var = y_sq_sum / len(y) - (y_sum ** 2) / len(y)**2

# Sigma de xy
xy_dev = ((x * y).sum() / len(x)) - (x_avg * y_avg)

# Coeficientes de la recta de regresión Pendiente y Ordenada
b1 = xy_dev / x_var
b0 = y_avg - b1 * x_avg

# Estimación de mínimos cuadrados
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
plt.title('Value vs ' + relevant_feature + 'con Recta de Regresión')
plt.xlabel(relevant_feature)
plt.ylabel('Value Normalized')
plt.legend()
plt.grid()
plt.show()

# i) Prueba de significancia de regresión, coeficiente de determinación (R²)
# y correlación lineal (r).

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
print("Coeficiente de determinacion: " + str(R2 * 100) + "%")
print("Coeficiente de correlación: " + str(r) + "\n")

# Prueba de significancia
# Se calcula el error estandar de la pendiente
dflen = len(x) - 2

sse = ((y - y_hat) ** 2).sum()
error_estandar_pendiente = sqrt(sse / (dflen * Sxx))

# Test de Hipótesis
t = b1_hat / error_estandar_pendiente

# Valor de P
p = 2 * (1 - stats.t.cdf(abs(t), dflen))

print("Error estandar de la pendiente: " + str(error_estandar_pendiente))
print("Test de Hipótesis: " + str(t))

# ii) Inferencias sobre los parámetros de la recta, estimando las
# fluctuaciones con una confianza del 95%.
# Conclusión del test
if p < 0.05:
    print("Se rechaza la hipótesis nula")
else:
    print("No se rechaza la hipótesis nula")

# iii) La proporción de veces que el valor de mercado supera la incertidumbre
# de predicción comparada con la respuesta media del valor de mercado para una
# característica fija, ambas con la misma confianza y ancho mínimo.

# Parámetros
alpha = 0.05
t_alpha_half = stats.t.ppf(1 - alpha / 2, dflen)  # Valor crítico de t
x0 = x.mean()  # Característica fija, puedes cambiar a otro valor específico

# Predicción para x0
y0_hat = b0_hat + b1_hat * x0

# Incertidumbre de predicción
pred_interval = t_alpha_half * \
    sqrt(sigma_hat_sqrd * (1 + 1 / len(x) + (x0 - x_avg)**2 / Sxx))

# Intervalo de confianza para la media
conf_interval = t_alpha_half * \
    sqrt(sigma_hat_sqrd / len(x) + (x0 - x_avg)**2 / Sxx)

# Calcular la proporción de veces que el valor de mercado supera la
# incertidumbre de predicción
exceeds_prediction_interval = np.sum(
    (y > (y0_hat + pred_interval)) | (y < (y0_hat - pred_interval)))
proportion_exceeds = exceeds_prediction_interval / len(y)

# Imprimir resultados
print(f"Intervalo de Predicción: ±{pred_interval}")
print(f"Intervalo de Confianza para la Media: ±{conf_interval}")
print(
    f"Proporción de veces que el valor de mercado supera la incertidumbre de predicción: {proportion_exceeds:.2f}")
