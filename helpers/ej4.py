from numpy import sqrt
import pandas as pd

def calcular_coeficiente_correlacion(x, y):
    # Calcular la sumatoria de las variables x e y
    x_sum = x.sum()
    y_sum = y.sum()

    # Calcular la media de las columnas 2 y 3
    x_avg = x.mean()
    y_avg = y.mean()

    # Calcular la sumatoria de los cuadrados de las variables x e y
    x_sq_sum = (x ** 2).sum()
    y_sq_sum = (y ** 2).sum()

    # Sigma al cuadrado de x e y (varianza)
    x_var = x_sq_sum / len(x) - (x_sum ** 2)
    y_var = y_sq_sum / len(y) - (y_sum ** 2)

    # Sigma de x e y (desviación estándar)
    #x_dev = sqrt(x_var)
    #y_dev = sqrt(y_var) 

    # Sigma de xy
    xy_dev = ((x * y).sum() / len(x)) - (x_avg * y_avg)

    # Coeficientes de la recta de regresión Pendiente y Ordenada
    b1 = xy_dev / x_var
    b0 = y_avg - b1 * x_avg 

    # Estimación de minimos cuadrados
    Sxy = (x * y).sum() - (x.sum() * y.sum()) / len(x)
    Sxx = (x ** 2).sum() - (x.sum() ** 2) / len(x)
    Syy = (y - y_avg).sum() ** 2

    b1_hat = Sxy / Sxx
    b0_hat = y_avg - b1_hat * x_avg
    y_hat = b1_hat * x + b0_hat

    # Cálculo de los residuos
    #residuals = y - y_hat

    # Calculo de Syy
    Syy = ((y - y_avg) ** 2).sum()

    # Suma de cuadrados de los residuos
    SSr = Syy - (b1_hat * Sxy)

    # Calculo de la varianza
    sigma_hat_sqrd = SSr / (len(x) - 2)

    # Calculo del coeficiente de determinación
    R2 = 1 - SSr/Syy
    r = Sxy / sqrt(Sxx * Syy)

    # Imprimir los resultados
    print("Estimación de la varianza: " + str(sigma_hat_sqrd))
    print("Coeficiente de determinación: " + str(R2))
    print("Probabilidad: " + str(R2 * 100) + "%")
    print("Coeficiente de correlación: " + str(r) + "\n")

#Lo se esta horrible, en otro momento lo arreglo jeje
def calcular_y(new_x, x, y):
    # Calcular la sumatoria de las variables x e y
    x_sum = x.sum()
    y_sum = y.sum()

    # Calcular la media de las columnas 2 y 3
    x_avg = x.mean()
    y_avg = y.mean()

    # Calcular la sumatoria de los cuadrados de las variables x e y
    x_sq_sum = (x ** 2).sum()
    y_sq_sum = (y ** 2).sum()

    # Sigma al cuadrado de x e y (varianza)
    x_var = x_sq_sum / len(x) - (x_sum ** 2)

    # Sigma de xy
    xy_dev = ((x * y).sum() / len(x)) - (x_avg * y_avg)

    # Coeficientes de la recta de regresión Pendiente y Ordenada
    b1 = xy_dev / x_var
    b0 = y_avg - b1 * x_avg 
        
    # Estimación de minimos cuadrados
    Sxy = (x * y).sum() - (x.sum() * y.sum()) / len(x)
    Sxx = (x ** 2).sum() - (x.sum() ** 2) / len(x)
    Syy = (y - y_avg).sum() ** 2

    b1_hat = Sxy / Sxx
    b0_hat = y_avg - b1_hat * x_avg

    print("Pendiente: " + str(b1_hat))
    print("Ordenada: " + str(b0_hat))

    print(b0_hat + b1_hat * new_x)


# Ruta del archivo Excel
ruta_archivo = 'TP2\ej4.xlsx'

# Leer el archivo Excel
df = pd.read_excel(ruta_archivo)

# Obtener los valores de las columnas 2 y 3
columna1 = df.iloc[:, 0].values
columna2 = df.iloc[:, 1].values

#calcular_coeficiente_correlacion(columna1, columna2)

#calcular_y(6, columna1, columna2)

#Reducimos en un 10% el tiempo de windows
columna1 = columna1 * 0.9
calcular_coeficiente_correlacion(columna1, columna2)
