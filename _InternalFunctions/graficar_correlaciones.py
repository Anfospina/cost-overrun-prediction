import matplotlib.pyplot as plt
def graficar_correlaciones(correlaciones):
    """
    Genera un gráfico de barras con las correlaciones calculadas.
    
    Args:
        correlaciones (dict): Diccionario con las correlaciones de Pearson.
    
    Returns:
        None
    """
    variables = list(correlaciones.keys())
    valores = list(correlaciones.values())

    plt.figure(figsize=(6, 6))
    plt.bar(variables, valores, color='skyblue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('Variables Numéricas', fontsize=12)
    plt.ylabel('Coeficiente de Correlación de Pearson', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()