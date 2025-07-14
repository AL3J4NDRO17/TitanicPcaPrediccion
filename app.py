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
knn = joblib.load('knnModelTitanic.pkl')                   # tu modelo KNN

app.logger.debug('Scaler, PCA y KNN cargados correctamente.')

most_important_cols = ['Sex_male', 'Age', 'Fare', 'Pclass', 'Cabin']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form.get('gender')
        age = request.form.get('age', type=float)
        fare = request.form.get('fare', type=float)
        pclass = request.form.get('pclass', type=int)
        cabin = request.form.get('cabin', '').strip()

        if not gender or age is None or fare is None or not pclass:
            return jsonify({'error': 'Faltan datos requeridos'}), 400

        sex_male = 1 if gender == 'male' else 0
        cabin_bin = 1 if cabin else 0

        input_dict = {
            'Sex_male': [sex_male],
            'Age': [age],
            'Fare': [fare],
            'Pclass': [pclass],
            'Cabin': [cabin_bin]
        }
        input_df = pd.DataFrame(input_dict)
        app.logger.debug(f'Datos recibidos: {input_df}')

        # Escalar
        input_scaled = scaler.transform(input_df)

        # Aplicar PCA
        input_pca = pca.transform(input_scaled)

        # Predecir con KNN
        pred = knn.predict(input_pca)[0]

        categoria = 'Sobrevivió' if pred == 1 else 'No Sobrevivió'

        return jsonify({'categoria': categoria})

    except Exception as e:
        app.logger.error(f'Error en predicción: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
