import pandas as pd
from linear_regression_model import LinearRegressionModel
from cleaner import Cleaner

def read_user_input(file_path):
    # Read user input from a text file
    try:
        user_input = pd.read_csv(file_path, sep='\t', index_col=0)
        return user_input
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def main():
    # Specify the path to your CSV file
    file_path = 'path/to/your/data.csv'

    # Instantiate the Cleaner class to clean the data
    cleaner = Cleaner(file_path)
    cleaned_data = cleaner.clean()

    # Instantiate the LinearRegressionModel class and perform regression
    linear_model = LinearRegressionModel(cleaned_data)
    linear_model.perform_regression()

    # Read user input from a text file
    user_input_path = 'user_input.txt'
    user_input = read_user_input(user_input_path)

    if user_input is not None:
        # Predict user input
        prediction = linear_model.predict_user_input(user_input)

        # Save results in a text file
        output_file_path = linear_model.results('output.txt')

        print(f"Prediction for user input: ${prediction:,.2f}")
        print(f"Results saved in {output_file_path}")

if __name__ == "__main__":
    main()
