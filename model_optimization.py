import pandas as pd
from _ModelFunctions.create_model_directory import create_model_directory
from _ModelFunctions.create_pipeline import create_pipeline
from _ModelFunctions.divide_project_data import divide_project_data
from _ModelFunctions.metrics_calculate_from_file import metrics_calculate_from_file
from _ModelFunctions.perform_grid_search_cv import perform_grid_search_cv
from sklearn.ensemble import GradientBoostingRegressor

# lectura de la data
project_2025=pd.read_csv('Input_output/Data/DataFrame.csv')

# división de la data en entrenamiento y prueba

#Iteraciones (1-3)
x_train, x_test, y_train, y_test = divide_project_data(project_2025,
                                                       target='Cantidad Real Gastada',
                                                       test_size=0.2,
                                                       random_state=42)

#Iteraciones (4-5)
x_train, x_test, y_train, y_test = divide_project_data(project_2025,
                                                       target='Cantidad Real Gastada',
                                                       test_size=0.2,
                                                       random_state=17)


#preprocesamiento de la data
pipeline_GB=create_pipeline(GradientBoostingRegressor(), model_name='GradientBoostingRegressor')

#Primera iteración
param_grid_GB = {
    'regressor__n_estimators': [100, 200, 300, 400],  
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2], 
    'regressor__max_depth': [3, 5, 7],  
    'regressor__min_samples_split': [2, 5, 10],  
    'regressor__min_samples_leaf': [1, 2, 4],  
    'regressor__subsample': [0.8, 1.0],  
    'regressor__max_features': ['sqrt', 'log2'] 
}

model_GB=perform_grid_search_cv(pipeline_GB, 
                       param_grid_GB, 
                       cv=8, 
                       scoring='neg_mean_absolute_error')

#Segunda iteración
param_grid_GB = {
    'regressor__n_estimators': [100, 200, 300, 400],  
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2], 
    'regressor__max_depth': [3, 5, 7],  
    'regressor__min_samples_split': [2, 5, 10],  
    'regressor__min_samples_leaf': [1, 2, 4],  
    'regressor__subsample': [0.8, 1.0],  
    'regressor__max_features': ['sqrt', 'log2'] 
}

model_GB=perform_grid_search_cv(pipeline_GB, 
                       param_grid_GB, 
                       cv=10, 
                       scoring='neg_mean_absolute_error')

#Tercera iteración
param_grid_GB = {
    'regressor__n_estimators': [50, 100, 150],  
    'regressor__learning_rate': [0.01, 0.05, 0.1],  
    'regressor__max_depth': [3, 5],  
    'regressor__min_samples_split': [5, 10],  
    'regressor__min_samples_leaf': [2, 4],  
    'regressor__subsample': [0.8, 1.0], 
    'regressor__max_features': ['sqrt']  
}

model_GB=perform_grid_search_cv(pipeline_GB, 
                       param_grid_GB, 
                       cv=8, 
                       scoring='neg_mean_absolute_error')

#Cuarta iteración
param_grid_GB = {
    'regressor__n_estimators': [150, 250, 350],  
    'regressor__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1], 
    'regressor__max_depth': [2, 4, 6],  
    'regressor__min_samples_split': [3, 4, 6],  
    'regressor__min_samples_leaf': [1, 3, 5],  
    'regressor__subsample': [0.7, 0.85, 1.0],  
    'regressor__max_features': ['sqrt', 'log2', 0.5] 
}

model_GB=perform_grid_search_cv(pipeline_GB, 
                       param_grid_GB, 
                       cv=8, 
                       scoring='neg_mean_absolute_error')

#Quinta iteración
param_grid_GB = {
    'regressor__n_estimators': [150, 250, 350, 400],  
    'regressor__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1], 
    'regressor__max_depth': [2, 4, 6],  
    'regressor__min_samples_split': [3, 4, 6],  
    'regressor__min_samples_leaf': [1, 3, 5],  
    'regressor__subsample': [0.7, 0.85, 1.0],  
    'regressor__max_features': ['sqrt', 'log2', 0.5] 
}

model_GB=perform_grid_search_cv(pipeline_GB, 
                       param_grid_GB, 
                       cv=8, 
                       scoring='neg_mean_absolute_error')

#ajustar el modelo con los mejores hiperparámetros
model_GB.fit(x_train, y_train)
print("Best score: ", model_GB.best_score_)
print("Best parameters: ", model_GB.best_params_)

#guardar el modelo entrenado
create_model_directory(model_GB, model_name='GradientBoosting_1.pkl.gz')

create_model_directory(model_GB, model_name='GradientBoosting_2.pkl.gz')

create_model_directory(model_GB, model_name='GradientBoosting_3.pkl.gz')

create_model_directory(model_GB, model_name='GradientBoosting_4.pkl.gz')

create_model_directory(model_GB, model_name='GradientBoosting_5.pkl.gz')

#cargar el modelo guardado
model_path_GB='optimization/GradientBoosting_1.pkl.gz'

model_path_GB='optimization/GradientBoosting_2.pkl.gz'

model_path_GB='optimization/GradientBoosting_3.pkl.gz'

model_path_GB='optimization/GradientBoosting_4.pkl.gz'

model_path_GB='optimization/GradientBoosting_5.pkl.gz'

#caculo de metricas del modelo
metrics_calculate_from_file(model_path_GB, x_train, x_test, y_train, y_test,
                            output_path='optimization/metrics1.json')

metrics_calculate_from_file(model_path_GB, x_train, x_test, y_train, y_test,
                            output_path='optimization/metrics2.json')

metrics_calculate_from_file(model_path_GB, x_train, x_test, y_train, y_test,
                            output_path='optimization/metrics3.json')

metrics_calculate_from_file(model_path_GB, x_train, x_test, y_train, y_test,
                            output_path='optimization/metrics4.json')

metrics_calculate_from_file(model_path_GB, x_train, x_test, y_train, y_test,
                            output_path='optimization/metrics5.json')
