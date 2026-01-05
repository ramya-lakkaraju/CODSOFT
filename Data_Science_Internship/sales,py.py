# Sales Prediction using Machine Learning
# CodSoft Data Science Internship - Task 4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the dataset
df = pd.read_csv("advertising.csv")

# 2. Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# 3. Dataset information
print("\nDataset Information:")
print(df.info())

# 4. Select features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 5. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict sales
y_pred = model.predict(X_test)

# 8. Evaluate the model
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
