import pandas as pd
import os
import gzip
import pickle
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    median_absolute_error)

def load_model_from_gz(file_path):
    with gzip.open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def metrics_calculate_from_file(model_path, x_train, x_test, y_train, y_test, output_path):
    # Cargar el modelo
    model = load_model_from_gz(model_path)

    # Realizar predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calcular métricas
    metrics = [
        {
            'type': 'metrics',
            'dataset': 'train',
            'r2': r2_score(y_train, y_train_pred),
            'mse': mean_squared_error(y_train, y_train_pred),
            'mad': median_absolute_error(y_train, y_train_pred),
        },
        {
            'type': 'metrics',
            'dataset': 'test',
            'r2': r2_score(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'mad': median_absolute_error(y_test, y_test_pred),
        }
    ]

    # Crear directorio de salida y guardar resultados
    os.makedirs('files/optimization', exist_ok=True)
    with open(output_path, 'w') as file:
        for metric in metrics:
            file.write(json.dumps(metric) + '\n')

# lectura de la data
project_2025=pd.read_csv('Archivos/Data/DataFrame.csv')

# preprocesamiento de la data
numerical_features=['Gasto a la Fecha Estimado', 'Presupuesto Final Estimado',
                    'Presupuesto del Proyecto','Probabilidad','Impacto','Duración Proyectada',
                    'Desviación Presupuestaria']
categorical_features=['Tipo Proyecto','Fase del Proyecto']

preprocessor=ColumnTransformer(
    transformers=
    [('num', StandardScaler(),numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# creación de la pipeline
steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
]
pipeline=Pipeline(steps)

param_grid = {
    'regressor__n_estimators': [150, 250, 350, 400],  
    'regressor__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1], 
    'regressor__max_depth': [2, 4, 6],  
    'regressor__min_samples_split': [3, 4, 6],  
    'regressor__min_samples_leaf': [1, 3, 5],  
    'regressor__subsample': [0.7, 0.85, 1.0],  
    'regressor__max_features': ['sqrt', 'log2', 0.5] 
}

# búsqueda de cuadrícula
model=GridSearchCV(pipeline,
                   param_grid=param_grid,
                   cv=8,
                   scoring='neg_mean_squared_error',
                   n_jobs=-1,
                   refit=True)

# partición de la data en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test=train_test_split(
    project_2025.drop(columns=['Cantidad Real Gastada']),
    project_2025['Cantidad Real Gastada'],
    test_size=0.2,
    random_state=17
)

# ajuste del modelo
model.fit(x_train, y_train)

# alamacenamiento del modelo
model_filename='files/optimization'
os.makedirs(model_filename,exist_ok=True)

model_path=os.path.join(model_filename,'GradientBoosting_5.pkl.gz')
with gzip.open(model_path,'wb') as file:
    pickle.dump(model,file)
    
model_path='files/optimization/GradientBoosting_5.pkl.gz'

# caculo de metricas del modelo
metrics_calculate_from_file(model_path, x_train, x_test, y_train, y_test,
                            output_path='files/optimization/metrics_5.json')