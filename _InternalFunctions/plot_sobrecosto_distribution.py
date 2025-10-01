import matplotlib.pyplot as plt
def plot_sobrecosto_distribution(df, column, title='Estado de Presupuestos'):
    """
    Genera un gráfico de barras para mostrar la distribución de proyectos con y sin sobrecosto.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        column (str): Nombre de la columna que contiene los valores de sobrecosto.
        title (str): Título del gráfico. Por defecto, 'Estado de Presupuestos'.

    Returns:
        None
    """
    # Clasificar los proyectos en 'Sin Sobrecosto' y 'Con Sobrecosto'
    conteo = df[column].apply(lambda x: 'Sin Sobrecosto' if x <= 0 else 'Con Sobrecosto').value_counts()

    # Crear el gráfico de barras
    conteo.plot(kind='bar', color=['blue', 'red'], rot=0)

    # Personalizar el gráfico
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('Cantidad de Proyectos')
    plt.gca().spines["top"].set_visible(False)  # Ocultar borde superior
    plt.gca().spines["right"].set_visible(False)  # Ocultar borde derecho
    plt.show()