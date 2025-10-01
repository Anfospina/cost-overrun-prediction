import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram_subplots(df, columns, title='Distribución de Presupuestos y Gastos', bins=30, log_scale=True):
    """
    Genera subgráficos (subplots) con histogramas individuales para las columnas seleccionadas de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columns (list): Lista de nombres de columnas para las que se generará el histograma.
        title (str): Título general del conjunto de gráficos.
        bins (int): Número de bins para el histograma. Por defecto, 30.
        log_scale (bool): Si True, aplica escala logarítmica al eje Y. Por defecto, True.

    Returns:
        None
    """
    n = len(columns)
    n_cols = 2  # Número de columnas de subgráficos (ajustable)
    n_rows = (n + n_cols - 1) // n_cols  # Calcular número de filas necesario

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # Asegurar que podemos indexar los ejes fácilmente

    for i, column in enumerate(columns):
        sns.histplot(data=df, x=column, bins=bins, kde=False, ax=axes[i], color='skyblue')
        axes[i].set_title(f'{column}')
        axes[i].set_xlabel('Valores')
        axes[i].set_ylabel('Frecuencia')
        if log_scale:
            axes[i].set_yscale('log')
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    # Eliminar gráficos vacíos si los hay
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()