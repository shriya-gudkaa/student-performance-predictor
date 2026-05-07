import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/marksheet.csv")

# Select subject columns
subjects = df.columns[5:]

# Create target (Pass/Fail)
df["Result"] = np.where(
    (df[subjects] < 35).any(axis=1),
    "Fail",
    "Pass"
)

# Features (inputs)
X = df[subjects]

# Target (output)
y = df["Result"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Test with new student
new_student = pd.DataFrame(
    [[85, 70, 90, 95]],
    columns=subjects
)

new_student = scaler.transform(new_student)
prediction = model.predict(new_student)
print("Prediction:", prediction[0])