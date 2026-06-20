"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit - Batch 2026

Goal:
    Build a basic classification model using a small dataset (Iris) and
    prove an understanding of the full supervised-learning pipeline.

Architecture (IPO Model, per the training deck):
    INPUT   -> Iris dataset (150 samples, 4 features, 3 classes) + Feature Scaling
    PROCESS -> Train/Test Split (shuffled) + K-Nearest Neighbors (KNN) algorithm
    OUTPUT  -> Confusion Matrix + Precision / Recall / F1 Score

Key Requirements covered:
    1. Load and understand a dataset      -> load_iris()
    2. Split data into train/test sets    -> train_test_split() (80/20, shuffled)
    3. Apply a simple classification algo -> KNeighborsClassifier (n_neighbors=5)
    4. Validate beyond raw accuracy       -> confusion matrix + classification report
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# PHASE 1: INPUT - Load & Understand the Dataset
# ---------------------------------------------------------------------------
def load_data():
    """Load the classic Iris benchmark dataset (150 samples, 4 features, 3 classes)."""
    iris = load_iris()
    X = iris.data            # features: sepal length/width, petal length/width
    y = iris.target          # labels: 0=setosa, 1=versicolor, 2=virginica
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names


# ---------------------------------------------------------------------------
# PHASE 2: PROCESS - Scale, Split, Train
# ---------------------------------------------------------------------------
def prepare_and_train(X, y, k=5, test_size=0.2, random_state=42):
    """
    1. Scale features (StandardScaler: mean=0, variance=1) so no single
       feature dominates the distance calculation used by KNN.
    2. Shuffle + split into train/test sets (removes order bias).
    3. Instantiate, fit, and return the trained KNN model.
    """
    # --- Feature Scaling (The Gatekeeper Rule) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train/Test Split (Structural Integrity) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # --- The Workflow: scikit-learn (Instantiate -> Fit -> Predict) ---
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    return model, scaler, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# PHASE 3: OUTPUT - Predict & Validate
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test, target_names):
    """Run predictions and print accuracy, confusion matrix, and F1 report."""
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names)

    print("=" * 55)
    print(" PROJECT 2: DATA CLASSIFICATION USING AI - RESULTS")
    print("=" * 55)
    print(f"\nTest set size: {len(y_test)} samples")
    print(f"Accuracy: {acc * 100:.2f}%\n")

    print("Confusion Matrix")
    print("(rows = actual class, columns = predicted class)")
    header = "            " + "  ".join(f"{n[:9]:>9}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{target_names[i][:9]:>9} " + "  ".join(f"{v:>9}" for v in row))

    print("\nClassification Report (Precision / Recall / F1-score)")
    print(report)

    return acc, cm, report


# ---------------------------------------------------------------------------
# PHASE 4: LIVE PREDICTION - Classify a brand-new flower
# ---------------------------------------------------------------------------
def predict_new_sample(model, scaler, target_names, sample):
    """
    Classify a brand-new, unseen flower measurement.
    sample = [sepal_length, sepal_width, petal_length, petal_width] in cm
    """
    sample_scaled = scaler.transform([sample])
    pred_class = model.predict(sample_scaled)[0]
    pred_proba = model.predict_proba(sample_scaled)[0]
    return target_names[pred_class], dict(zip(target_names, pred_proba))


if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(target_names)} classes -> {list(target_names)}\n")

    model, scaler, X_train, X_test, y_train, y_test = prepare_and_train(X, y, k=5)
    evaluate(model, X_test, y_test, target_names)

    # Demo: classify a brand-new, never-seen flower
    new_flower = [5.1, 3.5, 1.4, 0.2]  # looks like a setosa
    label, probs = predict_new_sample(model, scaler, target_names, new_flower)
    print(f"\nNew sample {new_flower} -> Predicted class: {label}")
    print(f"Confidence breakdown: {probs}")
