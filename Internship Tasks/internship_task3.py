# Fraud Detection Script
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv(r"D:\Internship Tasks\transactions.csv")  # correct file path
print("Dataset:\n", data)

# -----------------------------
# Step 1: Undersampling (balance classes)
# -----------------------------
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0].sample(len(fraud), random_state=42)  # match number of frauds
data_balanced = pd.concat([fraud, normal])

X = data_balanced[['Amount','Time']]  # features
y = data_balanced['Class']  # target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 2: Train Random Forest
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 3: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

print("\nPredictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
