import os
import joblib
import numpy as np

from src.contexts.api.models import PredictorRequest


class TrainModelController:
    def execute(self, request: PredictorRequest):
        print(request)
        email = request.email
        country = request.country
        city = request.city

        lr_model_path = os.getenv("MODELO_ENTRENADO")

        # Cargar el modelo y los encoders desde el archivo
        data = joblib.load(lr_model_path)
        modelo_cargado = data["model"]
        le_email = data["le_email"]
        le_country = data["le_country"]
        le_city = data["le_city"]
        le_genre = data["le_genre"]

        # Determinar el tipo de correo (mismo criterio usado al entrenar)
        if "@gmail.com" in email:
            email_type = "gmail"
        elif "@hotmail.com" in email:
            email_type = "hotmail"
        elif "@yahoo.com" in email:
            email_type = "yahoo"
        else:
            email_type = "other"

        # Codificar los datos de entrada con los mismos encoders del entrenamiento
        try:
            email_enc = le_email.transform([email_type])[0]
        except ValueError:
            email_enc = 0  # valor no visto en entrenamiento, usar default

        try:
            country_enc = le_country.transform([country])[0]
        except ValueError:
            country_enc = 0

        try:
            city_enc = le_city.transform([city])[0]
        except ValueError:
            city_enc = 0

        nuevo_dato = np.array([[email_enc, country_enc, city_enc]])

        # Hacer la predicción
        result = modelo_cargado.predict(nuevo_dato)
        genero_predicho = le_genre.inverse_transform(result)[0]

        print(f"Predicción de género musical: {genero_predicho}")

        return {"status": "OK", "result": genero_predicho}