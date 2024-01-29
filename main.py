import os
from clases import Cleaner, LinearRegressionModel
from werkzeug.utils import secure_filename
import pandas as pd
import requests
from io import StringIO

def read_user_input(file_path):
    # Read user input from a text file
    user_input_df = pd.read_csv(file_path)
    return user_input_df

def main():
    # Specify the path to your CSV file
    csv_file_path = 'data_today.csv'

    # Instantiate the Cleaner class to clean the data
    cleaner = Cleaner(csv_file_path)
    cleaned_data = cleaner.clean()

    # Instantiate the LinearRegressionModel class and perform regression
    linear_model = LinearRegressionModel(cleaned_data)
    linear_model.perform_regression()

    # Read user input from a text file
    user_input_file_path = 'data.csv'
    user_input = read_user_input(user_input_file_path)

    # Predict user input
    prediction = linear_model.predict_user_input(user_input)

    # Display results
    output_file_path = linear_model.results()

    print(f"Prediction for user input: ${prediction:,.2f}")
    print(f"Results saved in {output_file_path}")

if __name__ == "__main__":
    main()

