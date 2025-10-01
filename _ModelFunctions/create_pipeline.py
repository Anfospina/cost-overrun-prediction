def create_pipeline(estimator, model_name=None):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    numerical_features = [
        'Gasto a la Fecha Estimado', 'Presupuesto Final Estimado',
        'Presupuesto del Proyecto', 'Probabilidad', 'Impacto',
        'Duración Proyectada', 'Desviación Presupuestaria'
    ]
    categorical_features = ['Tipo Proyecto', 'Fase del Proyecto']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Si el modelo es LinearRegression, agrega SelectKBest (sin importar f_regression)
    if model_name == "LinearRegression":
        from sklearn.feature_selection import SelectKBest
        steps = [
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest()), 
            ('regressor', estimator)
        ]
    else:
        steps = [
            ('preprocessor', preprocessor),
            ('regressor', estimator)
        ]
    pipeline = Pipeline(steps)
    return pipeline
