from flask import Flask, request, render_template, send_from_directory
import os
from clases import Cleaner, LinearRegressionModel
from werkzeug.utils import secure_filename
import pandas as pd


model = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'archivos'

@app.route('/')
def upload():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file_post():
    file = request.files['file']
    if file.filename == '':
        return 'No file in the bucket'
    if file.filename.split('.')[-1] != 'csv':
        return 'The file must be a csv file'
    else:
        global model
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        model = file_path  # Update the global variable with file path
        return render_template('index.html', uploaded='Archivo subido')

model = model

@app.route('/clean')
def clean_and_show():
    global model
    if model != None:
        cleaned_file = Cleaner(file_path=model).clean()
        model = cleaned_file
        return render_template('index.html', cleaned_file='archivo limpio')
    else:
        return render_template('index.html', cleaned_file='no hay archivos')

model = model

@app.route('/feed')
def feed_the_model():
    global model
    model = LinearRegressionModel(data=model)
    model.perform_regression()
    return render_template('index.html', feed_model='El modelo ha sido entrenado')

model = model

@app.route('/predict', methods=['POST'])
def prediction():
    global model  # Assuming you have initialized the model somewhere in your code

    if model:
        pricem2 = request.form.get('pricem2', 'test')
        surface = request.form.get('surface', 'test')
        covered = request.form.get('covered', 'test')
        lat = request.form.get('lat', 'test')
        lon = request.form.get('lon', 'test')

    # You need to handle the case where model is not an instance of LinearRegressionModel
        user_predict = pd.DataFrame({'price_m2': [pricem2], 'surface': [surface], 'covered': [covered], 'lat': [lat], 'lon': [lon]})
        user_prediction = model.predict_user_input(user_predict)

        model.results()

        return render_template('index.html', userprediction=f'The house price would be: {user_prediction:,.2f} pesos') 
    else:
        return render_template('index.html', userprediction='TNo hay modelo para entranar') 
    

model = model

@app.route('/download')

def descarga():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output.txt', as_attachment=True)




if __name__ == "__main__":
    app.run(debug=True)