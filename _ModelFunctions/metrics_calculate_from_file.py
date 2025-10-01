from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    median_absolute_error)
import json
import os
import gzip
import pickle


def metrics_calculate_from_file(model_path, x_train, x_test, y_train, y_test, output_path):
    # Cargar el modelo
    with gzip.open(model_path, 'rb') as f:
         model = pickle.load(f)

    # Realizar predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calcular métricas
    metrics = [
        {
            'type': 'metrics',
            'dataset': 'train',
            'r2': r2_score(y_train, y_train_pred),
            'mse': mean_squared_error(y_train, y_train_pred),
            'mad': median_absolute_error(y_train, y_train_pred),
        },
        {
            'type': 'metrics',
            'dataset': 'test',
            'r2': r2_score(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'mad': median_absolute_error(y_test, y_test_pred),
        }
    ]

    # Crear directorio de salida y guardar resultados
    os.makedirs('files/output', exist_ok=True)
    with open(output_path, 'w') as file:
        for metric in metrics:
            file.write(json.dumps(metric) + '\n')