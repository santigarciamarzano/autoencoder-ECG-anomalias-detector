# Detección de Anomalías Cardíacas con Autoencoder

Este proyecto implementa un autoencoder para la detección de anomalías cardíacas en señales de ECG. Se entrena un modelo que reconstruye las señales normales, y las desviaciones en las señales anómalas se reflejan en mayores errores de reconstrucción, lo que permite identificar posibles anomalías.

## Descripción del Proyecto

El objetivo es entrenar un autoencoder para detectar anomalías cardíacas en señales de electrocardiogramas (ECG). El dataset utilizado contiene señales normalizadas de ECG con categorías que representan diferentes tipos de anomalías y señales normales.

Autoencoder: Se entrena para reconstruir señales normales y, posteriormente, se usa la pérdida de reconstrucción para detectar anomalías.
Preprocesamiento: Se normalizan las señales utilizando MinMaxScaler y se preparan para el modelo.
Visualización: Se incluyen gráficos de muestras de ECG, reconstrucciones y distribuciones de la pérdida de reconstrucción.
Métricas: Se utilizan la sensibilidad y especificidad para evaluar el rendimiento del modelo, priorizando la sensibilidad debido a la naturaleza de la tarea (detección de anomalías).

## Estructura del Proyecto

data/: Carpeta donde se almacenan los archivos CSV con los datos de entrenamiento y prueba (ECG5000).
scripts/: Módulos que contienen funciones de preprocesamiento, visualización, construcción del modelo y evaluación.
models/: Carpeta para almacenar los pesos del autoencoder entrenado.
requirements.txt: Lista de dependencias necesarias para ejecutar el proyecto.
app.py: Este script es para desplegar la aplicacióin en el entorno local con Streamlit

## Instrucciones

1- Instalar dependencias:

```bash

pip install -r requirements.txt
```
Ejecutar el notebook: El archivo principal en formato Jupyter Notebook contiene el flujo completo:
- Carga y preprocesamiento de datos
- Entrenamiento del autoencoder
- Evaluación y visualización de resultados

2- Entrenamiento y evaluación:
- El autoencoder se entrena utilizando señales normales del dataset
- Se evalúan las reconstrucciones para diferentes tipos de anomalías
- Se calcula un umbral para clasificar las señales reconstruidas y se visualizan las pérdidas y las métricas de sensibilidad/especificidad

3- Si quieres ejecutar la aplicación debes escribir la siguiente línea en el terminal con ubicación en el directorio raiz:

```bash
streamlit run app.py
```

## Resultados y Métricas

El modelo está optimizado para detectar las anomalías en las señales cardíacas, utilizando la pérdida de reconstrucción como métrica para distinguir entre señales normales y anómalas. La sensibilidad es la métrica principal, dado que es crucial identificar correctamente las anomalías, incluso a expensas de aumentar los falsos positivos.

## Notas Finales

- Este proyecto está diseñado para ser ligero y fácil de ejecutar. No es necesario descargar ningún dataset adicional, ya que los datos de entrenamiento y prueba están incluidos en el repositorio