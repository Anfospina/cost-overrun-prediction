import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import pickle
import pandas as pd

def plot_feature_importance_from_file(model_path):
    """
    Genera un gráfico de barras mostrando la importancia de las características
    de un modelo entrenado cargado desde un archivo.

    Args:
        model_path (str): Ruta al archivo del modelo entrenado (.pkl.gz).
    """
    # Cargar el modelo entrenado
    with gzip.open(model_path, "rb") as file:
        trained_model = pickle.load(file)

    # Verificar si el modelo tiene el atributo 'feature_importances_' o 'coef_'
    if hasattr(trained_model.best_estimator_.named_steps["regressor"], "feature_importances_"):
        # Modelos basados en árboles
        feature_importances = trained_model.best_estimator_.named_steps["regressor"].feature_importances_
    elif hasattr(trained_model.best_estimator_.named_steps["regressor"], "coef_"):
        # Modelos lineales
        feature_importances = trained_model.best_estimator_.named_steps["regressor"].coef_
    else:
        raise AttributeError("El modelo no tiene 'feature_importances_' ni 'coef_'.")

    # Obtener los nombres de las características
    preprocessor = trained_model.best_estimator_.named_steps["preprocessor"]
    categorical_features = preprocessor.transformers_[1][2]
    numerical_features = preprocessor.transformers_[0][2]

    # Obtener nombres codificados para variables categóricas
    encoded_categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)

    # Combinar nombres de características numéricas y categóricas
    all_feature_names = list(numerical_features) + list(encoded_categorical_features)

    # Crear un DataFrame para la importancia de las características
    importance_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    # Graficar la importancia de las características
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Importancia de las Características")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.tight_layout()
    plt.show()