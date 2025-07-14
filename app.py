from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Carga modelos: scaler, PCA y KNN
scaler = joblib.load('scalerTitanic.pkl')  # scaler para escalar features
pca = joblib.load('pcaModelTitanic.pkl')                   # tu PCA
model = joblib.load('knnModelTitanic.pkl')                   # tu modelo KNN

app.logger.debug('Scaler, PCA y KNN cargados correctamente.')

most_important_cols = ['Sex_male', 'Age', 'Fare', 'Pclass', 'Cabin']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir y limpiar datos
        gender = request.form.get('gender')
        age = float(request.form.get('age'))
        fare = float(request.form.get('fare'))
        pclass = int(request.form.get('pclass'))
        cabin_raw = request.form.get('cabin', '').strip()

        # Procesar cabina: primera letra o 'U' si vacío
        cabin_letter = cabin_raw[0].upper() if cabin_raw else 'U'
        cabin_num = ord(cabin_letter)  # Convierte letra a número ordinal

        # Convertir género a variable binaria
        sex_male = 1 if gender == 'male' else 0

        # Crear DataFrame con las columnas que tu modelo espera
        df = pd.DataFrame([[
            sex_male,
            age,
            fare,
            pclass,
            cabin_num
        ]], columns=['Sex_male', 'Age', 'Fare', 'Pclass', 'Cabin'])

        # Escalar
        df_scaled = scaler.transform(df)

        # Aplicar PCA
        df_pca = pca.transform(df_scaled)

        # Predecir
        prediction = model.predict(df_pca)

        # Convertir predicción a texto
        categoria = 'Sobrevivió' if prediction[0] == 1 else 'No Sobrevivió'

        return jsonify({'categoria': categoria})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)