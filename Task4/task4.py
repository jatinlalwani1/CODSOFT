# sales_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
data = pd.read_csv("advertising.csv")

# 2. Basic info and head
print("Dataset Head:\n", data.head())
print("\nDataset Info:\n")
print(data.info())

# 3. Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# 4. Visualize data
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(data=data, x='TV', y='Sales')
plt.title("TV vs Sales")

plt.subplot(1, 3, 2)
sns.scatterplot(data=data, x='Radio', y='Sales')
plt.title("Radio vs Sales")

plt.subplot(1, 3, 3)
sns.scatterplot(data=data, x='Newspaper', y='Sales')
plt.title("Newspaper vs Sales")

plt.tight_layout()
plt.show()

# 5. Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# 6. Define features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)

# 10. Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nR² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# 11. Display predicted vs actual
results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nSample Predictions:\n", results.head())

# 12. Visualize predictions
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.show()
