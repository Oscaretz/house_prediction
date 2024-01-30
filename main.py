import pandas as pd
from classes import Cleaner, LinearRegressionModel

def main():
    # Example of program application
    user_input_data = pd.read_csv('input.csv')
    
    # Creating an instance of Cleaner class
    cleaner = Cleaner('data_today.csv')
    
    # Cleaning the data
    cleaned_data = cleaner.clean()
    
    # Creating an instance of LinearRegressionModel class
    linear_regression = LinearRegressionModel(cleaned_data)
    
    # Performing regression
    linear_regression.perform_regression()
    
    # Predicting based on user input
    prediction = linear_regression.predict_user_input(user_input_data)
    
    # Saving results to CSV
    linear_regression.results(filename='output.csv')
    
    # Displaying the predicted price
    print('Predicted price: ${:,.2f}'.format(prediction))

if __name__ == "__main__":
    main()
