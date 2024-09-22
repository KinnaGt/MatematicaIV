import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("code/dataset/FIFA21.csv")

# Seleccionar las columnas de interés
y_key = 'value_eur'
x1_key = 'overall'
x2_key = 'potential'
df = df[[y_key, x1_key, x2_key]].dropna()

# Filtrar datos con overall mayor a 55
df = df[df[x1_key] > 55]
df = df[df[x1_key] < 80]

# Definir las variables
x1 = df[x1_key].values
x2 = df[x2_key].values
y = df[y_key].values

# Normalizar y a valores entre 0 y 100
y_min = y.min()
y_max = y.max()

y_normalized = 100 * (y - y_min) / (y_max - y_min)

# Imprimir estadísticas de la normalización
print(f"Min y: {y_min}, Max y: {y_max}")
print(f"y_normalized (primeros 5 valores): {y_normalized[:5]}")

# Calcular la suma de los cuadrados de x1 y x2
x1_sum = x1.sum()
x2_sum = x2.sum()
y_sum = y_normalized.sum()

x1_square = (x1 ** 2).sum()
x2_square = (x2 ** 2).sum()

x1_x2 = (x1 * x2).sum()

x1_y = (x1 * y_normalized).sum()
x2_y = (x2 * y_normalized).sum()

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
print(f"Coeficiente b2 (Potential): {b2}")

# Predicción usando la ecuación de regresión múltiple
y_hat_normalized = b0 + b1 * x1 + b2 * x2

# Desnormalizar las predicciones para compararlas con el rango original
y_hat = y_min + (y_hat_normalized / 100) * (y_max - y_min)

# Imprimir predicciones
print(f"Predicciones: {y_hat[:5]}")  # Mostrar solo las primeras 5 predicciones

# Imprimir la ecuación de regresión múltiple
print(f"Ecuación de regresión múltiple: y_hat = {b0} + {b1}*x1 + {b2}*x2")


# Crear el gráfico de dispersión
plt.scatter(y, y_hat, color='blue', label='Predicciones vs Valores Reales')

# Añadir la línea de referencia de "predicción perfecta"
min_val = min(y.min(), y_hat.min())
max_val = max(y.max(), y_hat.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Línea Ideal (y = y_hat)')

# Etiquetas del gráfico
plt.xlabel('Valor Real (y)')
plt.ylabel('Valor Predicho (y_hat)')
plt.title('Regresión Lineal Múltiple: Predicción vs Real')

# Mostrar leyenda
plt.legend()

# Mostrar gráfico
plt.grid()
plt.show()

# Calcular R²
ss_res = np.sum((y - y_hat) ** 2)  # Suma de los residuos al cuadrado
ss_tot = np.sum((y - y.mean()) ** 2)  # Suma total de los cuadrados
r2 = 1 - (ss_res / ss_tot)

print(f"Coeficiente de determinación (R²): {r2}")

corr_overall = np.corrcoef(x1, y)[0, 1]
corr_potential = np.corrcoef(x2, y)[0, 1]

print(f"Correlación entre overall y value_eur: {corr_overall}")
print(f"Correlación entre potential y value_eur: {corr_potential}")


# *ii) Usando el método de descenso por gradiente. ¿Son los valores obtenidos iguales a los
# * conseguidos mediante la resolución del sistema de ecuaciones normales? Muestra los
# * resultados obtenidos junto con las últimas iteraciones del algoritmo. Indica los valores de los
# * parámetros utilizados (como tasa de aprendizaje y número de iteraciones)


# * iii) Da una interpretación del criterio de corte utilizado en el algoritmo del gradiente. Explica
# * si presenta alguna falla. Si no es una buena condición de corte, ¿puedes sugerir un criterio
# * alternativo más eficaz?
