import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("dataset/FIFA21.csv")

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

# Armar matriz nβ0 + β1Sumatoria x1i + β2S umatoriax 2i = Sumatoria yi
# β0Sumatoriax1i + β1 Sumatoriax1i 2 + β2Sumatoria x1ix2i = Sumatoria x1iyi
# β0Sumatoriax2i + β1β0Sumatoriax2ix1ix2i + β2β0Sumatoriax2ix2i 2 = β0Sumatoriax2ix2iyi

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

# Calcular la Suma de los Cuadrados de los Residuos (SSR)
SSR = np.sum((y - y_hat) ** 2)

# Número total de observaciones
n = len(y)

# Número de variables independientes
k = 2

# Calcular la varianza residual (σ²)
sigma_squared = SSR / (n - k - 1)

# Imprimir los resultados
print(f"SSR: {SSR}")
print(f"Varianza Residual (σ²): {sigma_squared}")

# Calcular la Suma Total de los Cuadrados (SST)
y_avg = y.mean()
SST = np.sum((y - y_avg) ** 2)

# Calcular el coeficiente de determinación (R²)
R_squared = 1 - (SSR / SST)

# Calcular el coeficiente de determinación ajustado (Ra²)
R_adjusted_squared = 1 - (1 - R_squared) * (n - 1) / (n - k - 1)

# Calcular la raíz cuadrada del coeficiente de determinación ajustado (ra)
ra = np.sqrt(R_adjusted_squared)

# Imprimir los resultados
print(f"Coeficiente de Determinación (R²): {R_squared}")
print(f"Coeficiente de Determinación Ajustado (Ra²): {R_adjusted_squared}")
print(f"Raíz del Coeficiente de Determinación Ajustado (ra): {ra}")

# Gráfico de Predicciones vs Valores Reales
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y, y_hat, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')

# Gráfico de Residuos
residuos = y - y_hat
plt.subplot(1, 2, 2)
plt.scatter(y_hat, residuos, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuos vs Predicciones')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')

plt.tight_layout()
plt.show()
