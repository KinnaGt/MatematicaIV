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
df = df[df["overall"] > 65]

# Definir las variables
x1 = df[x1_key].values
x2 = df[x2_key].values
y = df[y_key].values

# Normalizar y a valores entre 0 y 100
y_min = y.min()
y_max = y.max()
y_normalized = 100 * (y - y_min) / (y_max - y_min)

# Añadir un sesgo (bias) de 1 a x1 y x2
X = np.column_stack((np.ones(len(x1)), x1, x2))
y = y_normalized

# Definir parámetros iniciales
theta = np.zeros(X.shape[1])  # Inicializamos los coeficientes en 0
alpha = 0.0001  # Tasa de aprendizaje
iterations = 1000  # Número de iteraciones

# Definir la función de costo
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

# Definir la función de descenso por gradiente
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        # Calcular las predicciones
        predictions = X.dot(theta)
        
        # Actualizar los coeficientes
        theta = theta - (alpha / m) * (X.T.dot(predictions - y))
        
        # Guardar el costo de cada iteración
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# Aplicar el descenso por gradiente
theta_optimal, cost_history = gradient_descent(X, y, theta, alpha, iterations)

# Imprimir los coeficientes finales
print(f"Coeficiente b0 (intersección): {theta_optimal[0]}")
print(f"Coeficiente b1 (Overall): {theta_optimal[1]}")
print(f"Coeficiente b2 (Potential): {theta_optimal[2]}")

# Predicción usando la ecuación de regresión múltiple
y_hat_normalized = X.dot(theta_optimal)

# Desnormalizar las predicciones para compararlas con el rango original
y_hat = y_min + (y_hat_normalized / 100) * (y_max - y_min)

# Imprimir predicciones
print(f"Predicciones: {y_hat[:5]}")  # Mostrar solo las primeras 5 predicciones

# Imprimir la ecuación de regresión múltiple
print(f"Ecuación de regresión múltiple: y_hat = {theta_optimal[0]} + {theta_optimal[1]}*x1 + {theta_optimal[2]}*x2")

# Imprimir el costo final
print(f"Costo final: {cost_history[-1]}")

# Calcular R²
ss_res = np.sum((y - y_hat_normalized) ** 2)  # Suma de los residuos al cuadrado
ss_tot = np.sum((y - y.mean()) ** 2)  # Suma total de los cuadrados
r2 = 1 - (ss_res / ss_tot)

print(f"Coeficiente de determinación (R²): {r2}")
