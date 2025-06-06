def perform_grid_search_cv(pipeline, param_grid, cv=10, scoring='neg_mean_squared_error'):
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True
    )
    return model