# Student Performance Predictor

This project predicts whether a student will **Pass or Fail** using three classification algorithms: K-Nearest Neighbors (KNN), Decision Tree, and Random Forest.

---

## Features

- Data preprocessing using pandas and NumPy
- Pass/Fail classification based on marks
- Three ML models implemented and compared: KNN, Decision Tree, and Random Forest
- Feature scaling using StandardScaler (for KNN)
- Feature importance analysis (Random Forest)
- Model evaluation using accuracy score and classification report
- Decision Tree visualization using matplotlib

---

## Model Comparison

| Model | Accuracy | Precision (Pass) | Recall (Pass) | F1-Score (Pass) | Needs Scaling |
|---|---|---|---|---|---|
| KNN | 90.67% | 0.94 | 0.73 | 0.82 | Yes |
| Decision Tree | 97.33% | 1.00 | 0.91 | 0.95 | No |
| Random Forest | 97.33% | 1.00 | 0.91 | 0.95 | No |

---

## What Improved from KNN → Decision Tree & Random Forest

| What Changed | KNN | Decision Tree & Random Forest |
|---|---|---|
| Accuracy | 90.67% | 97.33% (+6.66%) |
| Precision (Pass) | 0.94 | 1.00 |
| Recall (Pass) | 0.73 | 0.91 |
| F1-Score (Pass) | 0.82 | 0.95 |
| Feature Scaling | Required | Not needed |
| Interpretability | Hard to explain | Easy to visualize (tree diagram) |
| Feature Importance | Not available | Available (Random Forest) |
| Speed on large data | Slower (compares all points) | Faster (learns rules once) |

> **Key takeaway:** KNN struggled most with **Recall for Pass (0.73)** — meaning it missed 27% of students who actually passed and wrongly predicted them as Fail. Decision Tree and Random Forest fixed this, improving recall to 0.91 and overall accuracy by ~7%.

---

## Dataset

The dataset used in this project is taken from Kaggle.

- **Source:** Kaggle Student Marks Dataset
- Contains student details and marks in multiple subjects

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

---

## Models

### 1. K-Nearest Neighbors (KNN)
Classifies a student by looking at the K closest students in the dataset. Requires StandardScaler for feature normalization.

### 2. Decision Tree
Makes a series of yes/no decisions based on subject marks to classify Pass or Fail. Easy to visualize and interpret.

### 3. Random Forest
Builds 100 decision trees and combines their results for a more robust prediction. Also provides **feature importance** — showing which subject has the most impact on the result.

---

## Project Structure

```
student-performance-predictor/
│
├── data/
│   └── marksheet.csv
│
├── models/
│   ├── knn_model.py
│   ├── decision_tree.py
│   └── random_forest.py
│
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Run any model:
   ```
   python models/knn_model.py
   python models/decision_tree.py
   python models/random_forest.py
   ```

---

## Author

**Shriya Gudkaa** — [GitHub](https://github.com/shriya-gudkaa)
