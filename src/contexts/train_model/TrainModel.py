import numpy as np
import joblib
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class TrainModel:

    def entrenarModelo():

        #se usaron las credeciales para ingresar de manera ocacional (Transaction pooler)
        load_dotenv("/app/.env")
        USER = os.getenv("SUPABASE_USER")
        PASSWORD = os.getenv("SUPABASE_PASSWORD")
        HOST = os.getenv("SUPABASE_HOST")
        PORT = os.getenv("SUPABASE_PORT")
        DBNAME = os.getenv("SUPABASE_DBNAME")


        if(PORT== None):
            print("no se lee el env")
            return
        else:
            print("si se lee en env")


        try:
            with psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            ) as connection:
                with connection.cursor() as cursor:
                    # Consulta SQL - vista con email_type, country, city, genre
                    cursor.execute('SELECT email_type, country, city, genre FROM customer_genre_view;')
                    rows = cursor.fetchall()

                    print(f"Filas recuperadas: {len(rows)}")

        except Exception as e:
            print(f"Error al conectar o recuperar datos: {e}")
            return

        if not rows:
            print("No se recuperaron filas de la base de datos. Abortando entrenamiento.")
            return
        else:
            print(rows[:2])

        # Convertir a DataFrame para manejar columnas de texto fácilmente
        df = pd.DataFrame(rows, columns=["email_type", "country", "city", "genre"])

        # Codificar variables categóricas (texto -> números)
        le_email = LabelEncoder()
        le_country = LabelEncoder()
        le_city = LabelEncoder()
        le_genre = LabelEncoder()

        df["email_type_enc"] = le_email.fit_transform(df["email_type"])
        df["country_enc"] = le_country.fit_transform(df["country"])
        df["city_enc"] = le_city.fit_transform(df["city"])
        df["genre_enc"] = le_genre.fit_transform(df["genre"])

        x = df[["email_type_enc", "country_enc", "city_enc"]].values
        y = df["genre_enc"].values

        #dividir en entranamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        #entrenar el modelo (clasificación, no regresión)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(x_train, y_train)

        # Guardar el modelo Y los encoders juntos (los necesitamos para predecir después)
        joblib.dump({
            "model": model,
            "le_email": le_email,
            "le_country": le_country,
            "le_city": le_city,
            "le_genre": le_genre
        }, str(os.getenv("MODELO_ENTRENADO")))

        print("modelo entrenado")