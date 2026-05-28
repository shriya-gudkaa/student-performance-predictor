import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/marksheet.csv")

# Select subject columns
subjects = df.columns[5:]

# Create target (Pass/Fail) — same logic as KNN
df["Result"] = np.where(
    (df[subjects] < 35).any(axis=1),
    "Fail",
    "Pass"
)

# Features and target
X = df[subjects]
y = df["Result"]

# Train-test split (70-30) — same as KNN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


new_student = pd.DataFrame(
    [[85, 70, 90, 5]],
    columns=subjects
)

prediction = model.predict(new_student)
print("Prediction for new student:", prediction[0])

plt.figure(figsize=(20, 8))
plot_tree(
    model,
    feature_names=list(subjects),
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree - Student Performance")
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()
