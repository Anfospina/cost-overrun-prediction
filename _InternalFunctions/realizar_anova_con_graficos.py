import matplotlib.pyplot as plt
from scipy.stats import f_oneway
def realizar_anova_con_graficos(df, variables_categoricas, target):
    """
    Realiza ANOVA para múltiples variables categóricas contra un target numérico y genera un gráfico de P-Values.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        variables_categoricas (list): Lista de nombres de columnas categóricas.
        target (str): Nombre de la columna numérica objetivo.
    
    Returns:
        dict: Diccionario con los resultados de ANOVA (F-statistic y p-value) para cada variable categórica.
    """
    resultados = {}
    p_values = []
    for variable in variables_categoricas:
        grupos = [df[target][df[variable] == categoria] for categoria in df[variable].unique()]
        f_stat, p_value = f_oneway(*grupos)
        resultados[variable] = {'F-Statistic': f_stat, 'P-Value': p_value}
        p_values.append(p_value)

    # Crear gráfico de línea para los P-Values
    plt.figure(figsize=(8, 5))
    plt.plot(variables_categoricas, p_values, marker='o', color='blue', label='P-Value')
    plt.axhline(0.05, color='red', linestyle='--', linewidth=0.8, label='Nivel de Significancia (0.05)')
    plt.xlabel('Variables Categóricas', fontsize=12)
    plt.ylabel('P-Value', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return resultados