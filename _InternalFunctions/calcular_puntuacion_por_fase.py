def calcular_puntuacion_por_fase(dataframe, columna_fase, columna_probabilidad, columna_impacto, valor_fase):
    filtro = dataframe[dataframe[columna_fase] == valor_fase]
    media_probabilidad = filtro[columna_probabilidad].mean()
    media_impacto = filtro[columna_impacto].mean()
    return [media_probabilidad, media_impacto]