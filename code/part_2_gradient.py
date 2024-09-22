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

# Filtrar datos con overall mayor a 55 y menor a 80
df = df[df["overall"] > 55]
df = df[df["overall"] < 80]

# Verificar y limpiar datos
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Definir las variables
x1 = df[x1_key].values
x2 = df[x2_key].values
y = df[y_key].values

# Normalizar los datos manualmente
def normalize(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    normalized_X = (X - means) / stds
    return normalized_X, means, stds

# Agregar una columna de 1s para el término de intersección
X = np.vstack([x1, x2]).T
X, means, stds = normalize(X)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Añadir la columna de 1s

# Inicializar parámetros
theta = np.zeros(X.shape[1])
alpha = 0.01  # Ajuste de la tasa de aprendizaje
max_iterations = 1000
tolerance = 1e-6

# Función de costo
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Descenso por gradiente con criterio de convergencia
cost_history = []
for i in range(max_iterations):
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = (1 / len(y)) * X.T.dot(errors)
    
    # Debugging prints
    # print(f"Iteration {i+1}")
    # print(f"Predictions: {predictions[:5]}")
    # print(f"Errors: {errors[:5]}")
    # print(f"Gradient: {gradient}")
    
    theta -= alpha * gradient
    cost = compute_cost(X, y, theta)
    
    # Debugging cost
    # print(f"Cost: {cost}")

    cost_history.append(cost)
    
    # Verificar la convergencia
    if i > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
        print(f"Convergencia alcanzada después de {i+1} iteraciones")
        break

# Mostrar los resultados finales
print(f"Coeficiente b0 (intersección): {theta[0]}")
print(f"Coeficiente b1 (PesoIX1): {theta[1]}")
print(f"Coeficiente b2 (PesoAX2): {theta[2]}")

# Predicción usando la ecuación de regresión múltiple
y_hat = X.dot(theta)

# Imprimir predicciones
print(f"Predicciones: {y_hat[:5]}")  # Mostrar solo las primeras 5 predicciones
print(f"Ecuación de regresión múltiple: y_hat = {theta[0]} + {theta[1]}*x1 + {theta[2]}*x2")

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

plt.subplot(1, 3, 1)
plt.scatter(y, y_hat, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')

# Gráfico de Residuos
residuos = y - y_hat
plt.subplot(1, 3, 2)
plt.scatter(y_hat, residuos, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuos vs Predicciones')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.tight_layout()



# Gráfico de la función de costo durante las iteraciones
plt.subplot(1, 3, 3)

plt.plot(range(len(cost_history)), cost_history, color='blue')
plt.title('Historial de la Función de Costo')
plt.xlabel('Número de Iteración')
plt.ylabel('Costo')
plt.show()
