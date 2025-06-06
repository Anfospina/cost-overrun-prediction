import matplotlib.pyplot as plt
import seaborn as sns
def plot_boxplot(df, columns, title='Distribución de Duraciones y Retrasos'):
    """
    Genera un diagrama de caja (boxplot) para las columnas seleccionadas de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columns (list): Lista de nombres de columnas para las que se generará el boxplot.
        title (str): Título del gráfico. Por defecto, 'Distribución de Duraciones y Retrasos'.

    Returns:
        None
    """
    # Crear el DataFrame en formato largo para seaborn
    df_melted = df[columns].melt(var_name='Categoría', value_name='Valor')

    # Crear el diagrama de caja
    sns.catplot(x='Categoría', y='Valor', data=df_melted, kind='box')

    # Personalizar el gráfico
    plt.title(title)
    plt.xlabel('Categorías')
    plt.ylabel('Valores')
    plt.show()