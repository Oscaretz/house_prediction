# ... (Previous code)

class Cleaner:
    # ... (Previous code)

    def clean(self):
        # ... (Previous code)
        return self.houses3

class LinearRegressionModel:
    # ... (Previous code)

    def results(self, filename='output.txt'):
        with open(filename, 'w') as file:
            # Redirecting print statements to the file
            original_stdout = sys.stdout
            sys.stdout = file
            
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
            prediction = self.userprediction

            file.write('House to predict: \n')
            file.write(str(self.user_input_data) + '\n')
            file.write(f'\n-Mean Absolute Error: {mae}')
            file.write(f'\n-Mean Squared Error: {mse}')
            file.write(f'\n-Root Mean Squared Error: {rmse}')
            file.write(f'\n-R^2: {r_squared}')
            file.write(f'\nPredicted price based on user input: ${prediction:,.2f}')

            sys.stdout = original_stdout  # Resetting print output to console

        return filename

if __name__ == "__main__":
    main()
