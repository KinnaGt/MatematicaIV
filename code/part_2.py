import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv("code/dataset/test.csv")

# Seleccionar las columnas de interés
y_key = 'PesoFY'
x1_key = 'PesoIX1'
x2_key = 'PesoAX2'
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

print(A)
