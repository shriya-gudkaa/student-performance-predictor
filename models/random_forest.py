import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset
df = pd.read_csv("data/marksheet.csv")

# Select subject columns
subjects = df.columns[5:]

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

# Random Forest model (100 trees)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

print("\nFeature Importance (which subject matters most):")
for subject, score in zip(subjects, model.feature_importances_):
    print(f"  {subject}: {score:.4f}")


new_student = pd.DataFrame(
    [[85, 70, 90, 95]],
    columns=subjects
)

prediction = model.predict(new_student)
print("\nPrediction for new student:", prediction[0])

plt.figure(figsize=(20, 8))
plot_tree(
    model.estimators_[0],  # picks the 1st tree out of 100
    feature_names=list(subjects),
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("One Tree from Random Forest")
plt.savefig("random_forest_one_tree.png", dpi=150, bbox_inches="tight")
plt.show()
