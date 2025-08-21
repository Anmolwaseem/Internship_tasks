# Task 4: Predicting House Prices
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
data = pd.read_csv(r"D:\Internship Tasks\HousingData.csv")
print("Dataset preview:\n", data.head())

# -----------------------------
# Step 0: Handle missing values
# -----------------------------
data = data.fillna(data.mean())

# -----------------------------
# Step 1: Separate features and target
# -----------------------------
X = data.drop("MEDV", axis=1)  # features
y = data["MEDV"]               # target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 2: Split data and train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 3: Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nPredicted House Prices (first 10):\n", y_pred[:10])
print("\nRMSE:", rmse)


