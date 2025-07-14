from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado (asegúrate de poner el nombre correcto)
model = joblib.load('modelo_titanic.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')  # tu HTML debe llamarse así

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        gender = request.form['gender']       # 'male' o 'female'
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        pclass = int(request.form['pclass'])
        cabin = request.form.get('cabin', '')  # opcional

        # Procesar 'Sex_male' como variable dummy
        sex_male = 1 if gender == 'male' else 0

        # Procesar 'Cabin': puedes crear una variable binaria si tiene cabina o no
        cabin_flag = 0
        if cabin.strip() != '':
            cabin_flag = 1

        # Crear DataFrame con las columnas esperadas por el modelo
        data = pd.DataFrame([[
            sex_male,
            age,
            fare,
            pclass,
            cabin_flag
        ]], columns=['Sex_male', 'Age', 'Fare', 'Pclass', 'Cabin'])

        app.logger.debug(f'Data para predicción: \n{data}')

        # Predecir con el modelo
        prediction = model.predict(data)[0]

        # Mapear predicción a texto
        resultado = 'Sobrevivió' if prediction == 1 else 'No Sobrevivió'

        return jsonify({'categoria': resultado})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
