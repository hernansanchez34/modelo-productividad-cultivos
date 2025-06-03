# Simulación Inteligente de Productividad de Cultivos 🌱

Este proyecto implementa un sistema de simulación basado en modelos fisiológicos (WOFOST) y técnicas de aprendizaje automático para predecir el crecimiento y rendimiento de cultivos a partir de datos climáticos y del suelo.

## 🔧 Tecnologías utilizadas

- Python
- PCSE (WOFOST)
- scikit-learn
- XGBoost
- Prefect
- matplotlib
- pandas

## Preexploración

1. Utilizar creacion_dataset.ipynb para visualizar los datos con los que vamos a trabajar, este extrae los datos del directorio data/meteo

2. Para hacer lo mismo con los datos del .soil que tienen las características del terreno, debemos primero transformar el ".soil" a ".txt", para ello, utilizaremos el archivo converter.py

3. Ejecutaremos el visualizar_soil.ipynb para ver la percolación del terreno y sus características.

4. Crearemos el dataset de entrada con el archivo "creación_dataset.ipynb" para la posterior implementación del aprendizaje de máquina.

## 🚀 Estructura del flujo (orquestación.py)

1. Recolección de datos climáticos
2. Simulación de crecimiento con WOFOST
3. Entrenamiento de modelos (Random Forest, XGBoost)
4. Validación cruzada y comparación de métricas
5. Análisis de importancia de variables
6. Procesamiento en paralelo con Prefect 


