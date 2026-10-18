import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_model():

    # Load dataset
    data = pd.read_csv("student_data.csv")

    # Input features
    X = data[
        [
            "Attendance",
            "Study_Hours",
            "Previous_Score",
            "Assignment_Score",
            "Internal_Score"
        ]
    ]

    # Output / target
    y = data["Performance"]

    # Convert Good / Average / Poor into numbers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # -------------------------------------------------
    # TRAIN / TEST SPLIT
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -------------------------------------------------
    # RANDOM FOREST MODEL
    # -------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # -------------------------------------------------
    # TEST PREDICTION
    # -------------------------------------------------

    y_pred = model.predict(X_test)

    # -------------------------------------------------
    # CROSS-VALIDATION
    # -------------------------------------------------

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=scoring
    )

    # Average cross-validation scores
    accuracy = cv_results["test_accuracy"].mean()
    precision = cv_results["test_precision"].mean()
    recall = cv_results["test_recall"].mean()
    f1 = cv_results["test_f1"].mean()

    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    # -------------------------------------------------
    # STORE METRICS
    # -------------------------------------------------

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    return model, encoder, metrics