# Contenido completo del script con análisis de importancia de variables
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os
import time

@task
def cargar_datos():
    df = pd.read_csv("data/ml_csv/dataset_entrenamiento_clean.csv")
    X = df.drop(columns=["TWSO"])
    y = df["TWSO"]
    return X, y

@task
def definir_modelos():
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    xgb = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
    ])
    return {"Random Forest": rf, "XGBoost": xgb}

@task
def evaluar_un_modelo(nombre, modelo, X, y):
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    scorers = {
        "MAE": make_scorer(mean_absolute_error),
        "RMSE": make_scorer(rmse),
        "R2": make_scorer(r2_score)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = {m: cross_val_score(modelo, X, y, cv=kf, scoring=s, n_jobs=-1)
              for m, s in scorers.items()}
    return nombre, {m: (np.mean(val), np.std(val)) for m, val in scores.items()}

@task
def mostrar_resultados(resultados):
    df = pd.DataFrame(resultados).T
    if isinstance(df.iloc[0, 0], tuple):
        expanded_data = {}
        for model, metric_dict in resultados.items():
            for metric, (mean_val, std_val) in metric_dict.items():
                expanded_data.setdefault((metric, "mean"), {})[model] = mean_val
                expanded_data.setdefault((metric, "std"), {})[model] = std_val
        df = pd.DataFrame(expanded_data).sort_index(axis=1)
    print("\\nResumen de métricas:")
    print(df)

@task
def graficar_resultados(resultados):
    df = pd.DataFrame(resultados).T
    mean_df = df.applymap(lambda x: x[0] if isinstance(x, tuple) else x)
    output_dir = "resultados"
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        mean_df[metric].plot(kind="bar", ax=ax, color="cornflowerblue", edgecolor="black")
        ax.set_title(f"{metric} por modelo")
        ax.set_ylabel(metric)
        ax.set_xlabel("Modelo")
        ax.set_xticklabels(mean_df.index, rotation=0)
    plt.tight_layout()
    file_path = os.path.join(output_dir, "comparacion_metricas_subplots.png")
    plt.savefig(file_path, dpi=300)
    plt.show()
    print(f"\\n Gráfico guardado: {file_path}")

@task
def mostrar_importancia_variables(modelos, X, y):
    print("\\n Importancia de variables por modelo:")
    for nombre, pipeline in modelos.items():
        pipeline.fit(X, y)
        model = pipeline.named_steps["rf"] if "rf" in pipeline.named_steps else pipeline.named_steps["xgb"]
        importancias = model.feature_importances_
        variables = X.columns
        df = pd.DataFrame({"Variable": variables, "Importancia": importancias})
        df = df.sort_values("Importancia", ascending=False)
        print(f"\\n{nombre}\\n{df.to_string(index=False)}")  # <- CORREGIDO

@task
def graficar_importancia_variables(modelos, X, y):
    output_dir = "resultados"
    os.makedirs(output_dir, exist_ok=True)

    for nombre, pipeline in modelos.items():
        pipeline.fit(X, y)
        model = pipeline.named_steps["rf"] if "rf" in pipeline.named_steps else pipeline.named_steps["xgb"]
        importancias = model.feature_importances_
        variables = X.columns

        df = pd.DataFrame({"Variable": variables, "Importancia": importancias})
        df = df.sort_values("Importancia", ascending=False)

        # Crear el gráfico
        plt.figure(figsize=(10, 6))
        plt.barh(df["Variable"], df["Importancia"], color="steelblue", edgecolor="black")
        plt.xlabel("Importancia")
        plt.ylabel("Variable")
        plt.title(f"Importancia de Variables - {nombre}")
        plt.gca().invert_yaxis()  # Para que la variable más importante esté arriba

        # Guardar gráfico
        file_path = os.path.join(output_dir, f"importancia_variables_{nombre.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f" Gráfico de importancia guardado: {file_path}")



@flow(name="flujo_secuencial")
def flujo_entrenamiento_modelos_secuencial():
    start = time.time()

    X, y = cargar_datos()
    modelos = definir_modelos()
    resultados = {}

    for nombre, modelo in modelos.items():
        nombre_modelo, res = evaluar_un_modelo.fn(nombre, modelo, X, y)
        resultados[nombre_modelo] = res

    mostrar_resultados(resultados)
    graficar_resultados(resultados)
    mostrar_importancia_variables(modelos, X, y)
    graficar_importancia_variables(modelos, X, y)


    print(f"\\n Tiempo total (secuencial): {time.time() - start:.2f} segundos")

@flow(name="flujo_paralelo", task_runner=ConcurrentTaskRunner())
def flujo_entrenamiento_modelos_paralelo():
    start = time.time()

    X, y = cargar_datos()
    modelos = definir_modelos()

    tareas = [
        evaluar_un_modelo.submit(nombre, modelo, X, y)
        for nombre, modelo in modelos.items()
    ]
    resultados = {r.result()[0]: r.result()[1] for r in tareas}

    mostrar_resultados(resultados)
    graficar_resultados(resultados)
    mostrar_importancia_variables(modelos, X, y)
    graficar_importancia_variables(modelos, X, y)


    print(f"\\n Tiempo total (paralelo): {time.time() - start:.2f} segundos")

if __name__ == "__main__":
    flujo_entrenamiento_modelos_secuencial()
    flujo_entrenamiento_modelos_paralelo()
