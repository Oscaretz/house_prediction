#Importamos las bibliotecas a utilizar.
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import csv
from pymongo import MongoClient

'''
Cleaner, clase para recibir un archivo CSV y limpiarlo.
Funciones (2):
    - __init__: Recibir un archivo CSV.
    - clean: Limpiar la base de datos; regresa otra base de datos.
'''
class Cleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def clean(self):
        # Dropping duplicates
        self.df = self.df.drop_duplicates()

        # Reorganizing indices
        self.df["Unnamed: 0"] = range(1, len(self.df) + 1)
        self.df.set_index("Unnamed: 0", inplace=True)

        # Dropping NaNs based on 'price'
        self.df = self.df.dropna(subset=["price"])

        # Filtering for Distrito Federal
        self.df_distrito_federal = self.df[self.df['place_with_parent_names'].str.contains('Distrito Federal', na=False)]

        # Dropping NaNs in specific columns
        self.df_distrito_federal_no_nulos = self.df_distrito_federal.dropna(subset=['surface_total_in_m2', 'surface_covered_in_m2'], how='any')

        # Filtering 'DF' & 'MXN' in 'property_type' and 'currency'
        self.df_distrito_federal_no_nulos = self.df_distrito_federal_no_nulos[
            (self.df_distrito_federal_no_nulos['property_type'] == 'house') &
            (self.df_distrito_federal_no_nulos['currency'] == 'MXN')
        ]

        # Deleting characters in 'place_with_parent_names' column.
        self.df_distrito_federal_no_nulos['place_with_parent_names'] = self.df_distrito_federal_no_nulos['place_with_parent_names'].str.replace('|Distrito Federal|México|', '', regex=True)
        self.df_distrito_federal_no_nulos['place_with_parent_names'] = self.df_distrito_federal_no_nulos['place_with_parent_names'].str.replace('|', '', regex=True)

        # Selecting specific columns
        self.houses2 = self.df_distrito_federal_no_nulos[['place_with_parent_names', 'price_aprox_local_currency', 'price_per_m2', 'surface_total_in_m2', 'surface_covered_in_m2', 'lat-lon']]

        # Dropping rows with any NaN values
        self.houses2 = self.houses2.dropna()

        # Renaming columns
        self.houses2.rename(columns={
            'place_with_parent_names': 'place',
            'price_aprox_local_currency': 'price',
            'surface_total_in_m2': 'surface',
            'surface_covered_in_m2': 'covered',
            'price_per_m2': 'price_m2'
        }, inplace=True)

        # Changing the spaces in 'place' and spliting the column 'lat-lon'.
        self.houses2['place'] = self.houses2['place'].str.replace(' ', '_')
        self.houses2[['lat', 'lon']] = self.houses2['lat-lon'].str.split(pat=',', expand=True)
        self.houses2 = self.houses2.drop(columns='lat-lon')

        # Resetting DataFrame index.
        self.houses2.reset_index(drop=True, inplace=True)

        # Converting 'lat' and 'lon' to float.
        self.houses2['lat'] = self.houses2['lat'].astype(float)
        self.houses2['lon'] = self.houses2['lon'].astype(float)

        #---First outliers filter---
        datosnuevos = self.houses2.select_dtypes(include=['float64', 'int64'])

        predictores = datosnuevos[['price_m2', 'surface', 'covered']]
        target = datosnuevos['price']
        modelo_regresion = sm.OLS(target, sm.add_constant(predictores)).fit()

        leverage = pd.DataFrame(modelo_regresion.get_influence().hat_matrix_diag)

        umbral_leverage = 0.008
        filas_mayores_leverage = leverage[leverage > umbral_leverage].dropna().index

        outliers_test = modelo_regresion.outlier_test()
        filas_outliers_y = outliers_test[outliers_test['bonf(p)'] < 0.05].index

        indices_outliers = leverage.index.intersection([653,10,62,633,953, 74])

        filas_a_eliminar = set(filas_mayores_leverage).union(filas_outliers_y, indices_outliers)

        self.houses2 = datosnuevos.drop(filas_a_eliminar)

        #---Second outliers filter---
        columnas_a_quitar_outliers = ['price', 'price_m2', 'surface', 'covered', 'lat', 'lon']

        for columna in columnas_a_quitar_outliers:
            Q1 = self.houses2[columna].quantile(0.25)
            Q3 = self.houses2[columna].quantile(0.75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            self.houses3 = self.houses2[(self.houses2[columna] >= lower_limit) & (self.houses2[columna] <= upper_limit)]

        self.houses3.reset_index(drop=True, inplace=True)

        return self.houses3
'''
LinearRegressionModel, clase para entrenar el modelo de regresion lineal en base a la DB limpia 
para posteriormente aplicar la predicción.
Funciones(4):
    - __init__: Selecciona la base de datos a entrenar.
    - perform_regression: Entrena el modelo de la regresión lineal.
    - predict_user_input: Toma un df de casa a predecir; regresa métricas y el precio predicho.
    - results: Tomamos todos los outputs anteriores y los mandamos a un nuevo archivo de texto con todos los resultados.
'''
class LinearRegressionModel:
    def __init__(self, data):
        self.data = data

    def perform_regression(self):
        # Operations to fit the model
        X = self.data[['price_m2', 'surface', 'covered', 'lat', 'lon']]
        y = self.data['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r_squared = model.score(X_test, y_test)

        print('Mean Absolute Error:', mae)
        print('Mean Squared Error:', mse)
        print('Root Mean Squared Error:', rmse)
        print('R^2:', r_squared)

        self.model = model  # Storing the model
        
    def predict_all(self, data):
        # Usando el modelo entrenado para predecir el precio para todo el conjunto de datos
        predictions_all = self.model.predict(data[['price_m2', 'surface', 'covered', 'lat', 'lon']])
        # Agregar una nueva columna 'Predicted_Price' al DataFrame original con las predicciones
        data['Predicted_Price'] = predictions_all
        return data

    def predict_user_input(self, user_input):
        #Applying the trained model to predict based on the input
        user_prediction = self.model.predict(user_input)
        return user_prediction[0]
    
    def results(self, filename='output.csv'):
        with open(filename, 'w', newline='') as csv_file:
            # Write header to CSV file
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['MAE', 'MSE', 'RMSE', 'R_squared', 'Predicted_price'])

            X = self.data[['price_m2', 'surface', 'covered', 'lat', 'lon']]
            y = self.data['price']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions, squared=False)
            r_squared = model.score(X_test, y_test)
            prediction = self.predict_user_input(user_input_data)

            # Write results to CSV file
            csv_writer.writerow([mae, mse, rmse, r_squared, prediction])

            print(f"Results have been saved to '{filename}'.")


#---------------------------------------------------------------------






