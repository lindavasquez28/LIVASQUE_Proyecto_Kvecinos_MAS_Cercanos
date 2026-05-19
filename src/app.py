from utils import db_connect
engine = db_connect()

# your code here
# # LINDA VASQUEZ Proyecto K-vecinos más Cercanos

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://raw.githubusercontent.com/rosinni/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"
datos = pd.read_csv(url, sep=";")

def agrupar_calidad(calidad):
    if calidad <= 4:
        return 0  # Calidad baja
    elif calidad <= 6:
        return 1  # Calidad media
    else:
        return 2  # Calidad alta

datos['etiqueta'] = datos['quality'].apply(agrupar_calidad)

# ## División de los datos

x = datos.drop(["etiqueta", "quality"], axis=1)
y = datos["etiqueta"]

# (80/20)
x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(x, y, test_size=0.2, random_state=42)

escalador = StandardScaler()
x_entrenamiento_escalado = escalador.fit_transform(x_entrenamiento)
x_prueba_escalada = escalador.transform(x_prueba)

modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(x_entrenamiento_escalado, y_entrenamiento)

# primeras predicciones
predicciones = modelo_knn.predict(x_prueba_escalada)

print("Precisión inicial:", accuracy_score(y_prueba, predicciones))
print("\nMatriz de confusión:\n", confusion_matrix(y_prueba, predicciones))
print("\nReporte completo:\n", classification_report(y_prueba, predicciones))

# Buscar el mejor valor para k (probando del 1 al 20)
valores_k = range(1, 21)
lista_precisiones = []

for k in valores_k:
    modelo_prueba = KNeighborsClassifier(n_neighbors=k)
    modelo_prueba.fit(x_entrenamiento_escalado, y_entrenamiento)
    prediccion_k = modelo_prueba.predict(x_prueba_escalada)
    lista_precisiones.append(accuracy_score(y_prueba, prediccion_k))

plt.figure(figsize=(10, 5))
plt.plot(valores_k, lista_precisiones, marker='o')
plt.title("¿Qué K es mejor para adivinar?")
plt.xlabel("Valor de K")
plt.ylabel("Precisión")
plt.grid(True)
plt.show()

mejor_k = valores_k[lista_precisiones.index(max(lista_precisiones))]
print(f"\n¡El mejor valor de k que encontramos es: {mejor_k}!")

modelo_final = KNeighborsClassifier(n_neighbors=mejor_k)
modelo_final.fit(x_entrenamiento_escalado, y_entrenamiento)

# ## Función para jugar al sommelier

def predecir_calidad_vino(caracteristicas_vino):
    vino_escalado = escalador.transform([caracteristicas_vino])
    resultado = modelo_final.predict(vino_escalado)[0]
    
    if resultado == 0:
        return "Este vino probablemente sea de baja calidad"
    elif resultado == 1:
        return "Este vino probablemente sea de calidad media"
    else:
        return "Este vino probablemente sea de alta calidad"

# Prueba
respuesta = predecir_calidad_vino([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
print("\nPredicción de nuestro vino de prueba:", respuesta)