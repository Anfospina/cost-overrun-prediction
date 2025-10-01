import matplotlib.pyplot as plt
import gzip
import pickle
def plot_model_predictions(model_path, x_test, y_test):
    """
    Carga un modelo entrenado desde un archivo .pkl.gz y genera un gráfico de predicciones vs valores reales.

    Args:
        model_path (str): Ruta del archivo del modelo entrenado (.pkl.gz).
        x_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Valores reales del conjunto de prueba.

    Returns:
        None
    """
    # Cargar el modelo desde el archivo .pkl.gz
    with gzip.open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Realizar predicciones
    y_test_pred = model.predict(x_test)

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.3, label="Predicciones")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Línea Ideal")
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    plt.legend()
    plt.show()