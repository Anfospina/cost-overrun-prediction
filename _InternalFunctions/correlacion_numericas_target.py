import matplotlib.pyplot as plt
def correlacion_numericas_target(df, variables_numericas, target):
    """
    Calcula la correlación de Pearson entre variables numéricas y el target.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        variables_numericas (list): Lista de nombres de columnas numéricas.
        target (str): Nombre de la columna numérica objetivo.
    
    Returns:
        dict: Diccionario con las correlaciones de Pearson para cada variable numérica.
    """
    correlaciones = {}
    for variable in variables_numericas:
        correlacion = df[[variable, target]].corr(method='pearson').iloc[0, 1]
        correlaciones[variable] = correlacion
    return correlaciones
