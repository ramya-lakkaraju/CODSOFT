# Movie Rating Prediction using Linear Regression
# CodSoft Data Science Internship - Task 2

import pandas as pd
import numpy as np

# Load dataset (handle encoding issue)
df = pd.read_csv("movies.csv", encoding="latin1")

# View dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

# Select required columns (MATCHING DATASET)
df = df[['Genre', 'Director', 'Actor 1', 'Rating']]

# Handle missing values
df = df.dropna()

# Separate features and target
X = df[['Genre', 'Director', 'Actor 1']].copy()
y = df['Rating']

# Convert categorical columns to numerical using Label Encoding
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for col in X.columns:
    X[col] = encoder.fit_transform(X[col])

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Predict ratings
y_pred = model.predict(X_test)

# Evaluate model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
