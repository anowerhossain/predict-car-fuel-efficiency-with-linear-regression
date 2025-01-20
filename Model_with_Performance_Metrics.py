# Importing Required Libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.linear_model import LinearRegression  # For building the linear regression model
from sklearn.model_selection import train_test_split  # For splitting the data into training and testing sets
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # For evaluating the model
from math import sqrt  # For calculating the square root (used in RMSE)

# Task 1 - Load the data into a DataFrame
# URL where the dataset is available
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"

# Load the dataset into a DataFrame using pandas
mpg_data = pd.read_csv(dataset_url)

# Display a sample of 5 random rows from the dataset to understand the structure of the data
mpg_data.sample(5)

# Get the number of rows and columns in the dataset (392 rows and 8 columns)
mpg_data_shape = mpg_data.shape  # This returns (392, 8)

# Create a scatter plot to visualize the relationship between Weight and MPG (Miles per Gallon)
mpg_data.plot.scatter(x="Weight", y="MPG")

# Task 2 - Identify the target and feature columns
# The target column we want to predict is 'MPG' (Miles per Gallon)
target_column = mpg_data["MPG"]

# The features we will use to predict 'MPG' are 'Horsepower' and 'Weight'
feature_columns = mpg_data[["Horsepower", "Weight"]]

# Task 3 - Split the data into training and testing sets
# We split the data in a 70:30 ratio, where 70% will be used for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(feature_columns, target_column, test_size=0.30, random_state=42)

# Task 4 - Build and Train a Linear Regression Model
# Create the Linear Regression model
linear_regressor = LinearRegression()

# Train the model using the training data
linear_regressor.fit(X_train, y_train)

# Task 5 - Evaluate the model
# Use the testing data to evaluate the model's performance
model_score = linear_regressor.score(X_test, y_test)

# Calculate the predicted values based on the testing features
predicted_values = linear_regressor.predict(X_test)

# R-squared: Higher the value, better the model
r_squared = r2_score(y_test, predicted_values)

# Mean Squared Error: Lower the value, better the model
mse = mean_squared_error(y_test, predicted_values)

# Root Mean Squared Error: Lower the value, better the model
rmse = sqrt(mse)

# Mean Absolute Error: Lower the value, better the model
mae = mean_absolute_error(y_test, predicted_values)

# Output the model evaluation metrics
print(f"Model Score: {model_score}")
print(f"R-squared: {r_squared}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
