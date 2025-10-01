"""Web application to deploy the cost overrun prediction model"""

import pickle
import gzip
import pandas as pd  # type: ignore
from flask import Flask, render_template, request  # type: ignore


# es obligatorio crear esta variable para que Flask funcione
app = Flask(__name__)
app.config["SECRET_KEY"] = "cost-overrun-predictor-2025" # clave para proteger la sesión del usuario


@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=("GET", "POST"))
def index():
    """Main web page for cost overrun prediction"""

    if request.method == "POST":

        user_values = {}

        # Variables numéricas
        user_values["Gasto a la Fecha Estimado"] = float(request.form["gasto_fecha_estimado"])
        user_values["Presupuesto Final Estimado"] = float(request.form["presupuesto_final_estimado"])
        user_values["Presupuesto del Proyecto"] = float(request.form["presupuesto_proyecto"])
        user_values["Probabilidad"] = float(request.form["probabilidad"])
        user_values["Impacto"] = float(request.form["impacto"])
        user_values["Duración Proyectada"] = float(request.form["duracion_proyectada"])
        user_values["Desviación Presupuestaria"] = float(request.form["desviacion_presupuestaria"])

        # Variables categóricas
        user_values["Tipo Proyecto"] = request.form["tipo_proyecto"]
        user_values["Fase del Proyecto"] = request.form["fase_proyecto"]

        # Crear DataFrame con los valores del usuario
        df = pd.DataFrame.from_dict(user_values, orient="index").T

        # Cargar el modelo entrenado
        try:
            with gzip.open("optimization/GradientBoosting_4.pkl.gz", "rb") as file:
                loaded_model = pickle.load(file)
            
            # Realizar predicción
            prediction = round(loaded_model.predict(df)[0], 2)
        except FileNotFoundError:
            prediction = "Error: Modelo no encontrado"
        except Exception as e:
            prediction = f"Error en predicción: {str(e)}"

    else:
        prediction = None

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)