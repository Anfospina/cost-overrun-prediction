def clean_outliers(df, columns):

    project_clean = df.copy()
    outlier_indices = set()  # Conjunto para almacenar los índices de las filas con valores atípicos

    for column in columns:
        # Calcular Q1, Q3 e IQR
        q1 = project_clean[column].quantile(0.25)
        q3 = project_clean[column].quantile(0.75)
        iqr = q3 - q1

        # Calcular límites superior e inferior
        upper = q3 + (2 * iqr)
        lower = q1 - (2 * iqr)

        # Identificar índices de valores atípicos
        outliers = project_clean[(project_clean[column] > upper) | (project_clean[column] < lower)].index
        outlier_indices.update(outliers)  # Agregar los índices al conjunto

    # Eliminar todas las filas con valores atípicos
    project_clean = project_clean.drop(index=outlier_indices)
    return project_clean