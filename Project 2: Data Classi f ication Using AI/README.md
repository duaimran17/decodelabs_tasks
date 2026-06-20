# Project 2: Data Classification Using AI 🌸

**Track:** DecodeLabs Industrial Training Kit — Batch 2026
**Module:** Predictive Phase — Supervised Learning

## 📌 Goal
Build a basic classification model using a small dataset (the classic **Iris**
benchmark) — proving you can train, test, and validate an AI model.

## ✅ Requirements Covered
| Requirement | Implementation |
|---|---|
| Load and understand a dataset | `sklearn.datasets.load_iris()` — 150 samples, 4 features, 3 classes |
| Feature scaling | `StandardScaler` (mean=0, variance=1) before distance-based learning |
| Split into training/testing sets | `train_test_split()` — 80% train / 20% test, shuffled to remove order bias |
| Apply a classification algorithm | `KNeighborsClassifier(n_neighbors=5)` — K-Nearest Neighbors |
| Output validation | Accuracy, Confusion Matrix, Precision/Recall/F1 (`classification_report`) |

## 🧠 How It Works (IPO Model)
1. **Input** — Load the Iris dataset (sepal/petal length & width) and scale
   every feature so no single measurement dominates the distance calculation.
2. **Process** — Shuffle and split the data (80/20), then train a KNN model:
   for any new point, it looks at its **5 nearest neighbors** in the training
   set and takes a majority vote on the species.
3. **Output** — Predict on the unseen test set, then validate with more than
   just accuracy (which can be misleading on imbalanced data) — a confusion
   matrix and per-class F1 score show exactly where the model is strong.

## 📊 The Dataset
| Samples | Classes | Features |
|---|---|---|
| 150 (balanced, 50 per class) | setosa, versicolor, virginica | sepal length, sepal width, petal length, petal width (cm) |

## ▶️ How to Run
```bash
pip install scikit-learn numpy
python3 classification.py
```

## 💬 Example Output
```
Dataset loaded: 150 samples, 4 features, 3 classes -> ['setosa', 'versicolor', 'virginica']

Test set size: 30 samples
Accuracy: 100.00%

Confusion Matrix
(rows = actual class, columns = predicted class)
              setosa  versicolor  virginica
   setosa        10           0          0
versicolor         0           9          0
 virginica         0           0         11

Classification Report (Precision / Recall / F1-score)
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11
    accuracy                           1.00        30
```

> Iris is a famously "clean" dataset, so 100% accuracy here is expected and
> not a sign of overfitting — it's a good sanity check that the pipeline
> (scaling → split → fit → predict → validate) is wired correctly.

## 🛠 Possible Extensions
- Try different values of `k` and plot the "elbow" to find the optimal one
- Compare KNN against other algorithms (Logistic Regression, Decision Tree)
- Test the model on a deliberately noisy or imbalanced dataset
