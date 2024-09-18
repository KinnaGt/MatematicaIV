import numpy as np
from scipy import stats

# Funciones útiles

def avg_num(num):
    return num.mean()

def sum_sqrd(num):
    return ((num - avg_num(num)) ** 2).sum()

def get_min_sxx(x):
    return sum_sqrd(x)

def get_min_sxy(x, y):
    return ((x - avg_num(x)) * (y - avg_num(y))).sum()

def get_min_syy(y):
    return sum_sqrd(y)

def get_b1_hat(x, y):
    Sxy = get_min_sxy(x, y)
    Sxx = get_min_sxx(x)
    return Sxy / Sxx

def get_b0_hat(x, y):
    b1_hat = get_b1_hat(x, y)
    return avg_num(y) - b1_hat * avg_num(x)

def get_y_hat(x, y):
    b1_hat = get_b1_hat(x, y)
    b0_hat = get_b0_hat(x, y)
    return b1_hat * x + b0_hat

def error_estandar_pendiente(x, y):
    Sxx = get_min_sxx(x)
    SSE = ((y - get_y_hat(x,y)) **2).sum()
    return np.sqrt( SSE / ((len(x) - 2) * Sxx))


# Datos proporcionados
x = np.array([5, 10, 15, 20, 25, 10, 25, 10, 5, 10, 20, 5, 10, 25, 0])
y = np.array([9.6, 20.1, 29.9, 39.1, 50.0, 9.6, 19.4, 29.7, 40.3, 49.9, 10.7, 21.3, 30.7, 41.8, 51.2])


# Calculo de la estimación de la recta
"""
print("Valor de Sxx: " + str(get_min_sxx(x)))
print("Valor de Sxy: " + str(get_min_sxy(x, y)))
print("Valor de Syy: " + str(get_min_syy(y)))

print("Valor de la estimación de la pendiente (b1): " + str(get_b1_hat(x, y)))
print("Valor de la estimación de la ordenada (b0): " + str(get_b0_hat(x, y)))

"""
# Estimación de los valores de Y con la recta ajustada
y_estimada = get_y_hat(x, y)
#print("Valores estimados de y:", y_estimada)


# Test de Hipótesis
# H0: b1 = 0
# HA: b1 != 0
# Calculo de estadístico de prueba
t = get_b1_hat(x, y) / error_estandar_pendiente(x, y)

# Grados de libertad
df = len(x) - 2

# Valor de P
p = 2 * (1 - stats.t.cdf(abs(t), df))


# Conclusión del test
if p < 0.05:
    print("Se rechaza la hipótesis nula")
else:
    print("No se rechaza la hipótesis nula")
