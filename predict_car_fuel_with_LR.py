import warnings
import pandas as pd
from sklearn.linear_model import LinearRegression

# Suppress warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings("ignore")

# Install required libraries if not already installed
# Uncomment the lines below if running in an environment without pre-installed libraries.
# !pip install pandas==1.3.4
# !pip install scikit-learn==1.0.2
# !pip install numpy==1.21.6

# URL of the dataset
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"

# Load the dataset
df = pd.read_csv(URL)

# Display sample data
print("Sample data:")
print(df.sample(5))

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Scatter plot of Horsepower vs MPG
df.plot.scatter(x="Horsepower", y="MPG", title="Horsepower vs MPG")

# Identify target (dependent variable)
target = df["MPG"]

# Identify features (independent variables)
features = df[["Horsepower", "Weight"]]

# Create Linear Regression model
lr = LinearRegression()

# Train the model
lr.fit(features, target)
print("\nModel trained.")

# Evaluate the model
score = lr.score(features, target)
print(f"\nModel score (R^2): {score:.2f}")

# Make predictions
horsepower = 100
weight = 2000
prediction = lr.predict([[horsepower, weight]])
print(f"\nPredicted MPG for a car with Horsepower={horsepower} and Weight={weight}: {prediction[0]:.2f}")
