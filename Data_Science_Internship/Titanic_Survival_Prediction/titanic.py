# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 3. Display first 5 rows
print(df.head())

# 4. Dataset information
print(df.info())

# 5. Statistical summary
print(df.describe())

# 6. Visualize survival count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# 7. Convert 'Sex' column to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 8. Fill missing Age values with mean age
df['Age'].fillna(df['Age'].mean(), inplace=True)

# 9. Select input features and target variable
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
y = df['Survived']

# 10. Split dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 11. Train Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 12. Make predictions
y_pred = model.predict(X_test)

# 13. Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
