# ============================================================
# Student Performance Prediction System
# Model Training + Evaluation + Probability Calibration
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


warnings.filterwarnings("ignore")


# ============================================================
# 1. CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATA_PATH = "data/student_performance.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset loaded successfully!")
print(f"Number of records: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
print(list(df.columns))


# ============================================================
# 3. BASIC DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("\nMissing values:")
print(df.isnull().sum())

print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Remove duplicate rows if any
df = df.drop_duplicates().reset_index(drop=True)


# ============================================================
# 4. DEFINE TARGET AND FEATURES
# ============================================================

TARGET = "final_result"


# IMPORTANT:
# student_id is intentionally NOT used as a predictive feature.
# final_result is the target and is also NOT used as a feature.

FEATURES = [

    # Existing features
    "gender",
    "age",
    "attendance",
    "study_hours",
    "previous_percentage",
    "internal_marks",
    "assignment_marks",
    "practical_marks",
    "backlogs",
    "extracurricular",

    # New features
    "class_year",
    "semester",
    "number_of_subjects",
    "previous_exam_score",
    "project_presentation_marks",
    "quiz_test_scores",
    "study_consistency",
    "assignment_submission_rate",
    "learning_study_method",
    "final_exam_marks"
]


X = df[FEATURES]
y = df[TARGET]


print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(f"\nTotal input features: {len(FEATURES)}")

print("\nFeatures used for training:")
for feature in FEATURES:
    print("-", feature)

print("\nTarget:")
print(TARGET)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 6. PREPROCESSING
# ============================================================

# Categorical features
categorical_features = [
    "gender",
    "class_year",
    "study_consistency",
    "learning_study_method",
    "extracurricular"
]


# Numerical features
numerical_features = [
    "age",
    "attendance",
    "study_hours",
    "previous_percentage",
    "internal_marks",
    "assignment_marks",
    "practical_marks",
    "backlogs",
    "semester",
    "number_of_subjects",
    "previous_exam_score",
    "project_presentation_marks",
    "quiz_test_scores",
    "assignment_submission_rate",
    "final_exam_marks"
]


# ============================================================
# NUMERICAL PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# ============================================================
# COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 7. CREATE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    ),

    "Decision Tree": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                DecisionTreeClassifier(
                    random_state=42
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    ),

    "KNN": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                KNeighborsClassifier()
            )
        ]
    )
}


# ============================================================
# 8. HYPERPARAMETER GRIDS
# ============================================================

param_grids = {

    "Logistic Regression": {
        "model__C": [0.1, 1, 10],
        "model__solver": ["lbfgs"]
    },

    "Decision Tree": {
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2, 4]
    },

    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    },

    "KNN": {
        "model__n_neighbors": [5, 9, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2]
    }
}


# ============================================================
# 9. TRAIN AND TUNE MODELS
# ============================================================

results = {}
trained_models = {}


for model_name in models:

    print("\n" + "=" * 70)
    print(f"TUNING {model_name.upper()}")
    print("=" * 70)

    grid_search = GridSearchCV(
        estimator=models[model_name],
        param_grid=param_grids[model_name],
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    trained_models[model_name] = best_model

    print("\nBest parameters:")
    print(grid_search.best_params_)


# ============================================================
# 10. EVALUATE ALL MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)


for model_name, model in trained_models.items():

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print(f"\n{model_name}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(results_df)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = results_df["F1 Score"].idxmax()

best_uncalibrated_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)
print("BEST MODEL BEFORE CALIBRATION")
print("=" * 70)

print(f"Selected model: {best_model_name}")

print(
    f"Accuracy      : "
    f"{results_df.loc[best_model_name, 'Accuracy']:.4f}"
)

print(
    f"Precision     : "
    f"{results_df.loc[best_model_name, 'Precision']:.4f}"
)

print(
    f"Recall        : "
    f"{results_df.loc[best_model_name, 'Recall']:.4f}"
)

print(
    f"F1 Score      : "
    f"{results_df.loc[best_model_name, 'F1 Score']:.4f}"
)


# ============================================================
# 13. PROBABILITY CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY CALIBRATION")
print("=" * 70)

print("\nApplying sigmoid probability calibration...")

print(
    "This does NOT manually change probabilities."
)

print(
    "It recalibrates the model's probability estimates."
)


calibrated_model = CalibratedClassifierCV(
    estimator=best_uncalibrated_model,
    method="sigmoid",
    cv=5
)


calibrated_model.fit(
    X_train,
    y_train
)


# ============================================================
# 14. EVALUATE CALIBRATED MODEL
# ============================================================

calibrated_predictions = calibrated_model.predict(
    X_test
)


calibrated_accuracy = accuracy_score(
    y_test,
    calibrated_predictions
)


calibrated_precision = precision_score(
    y_test,
    calibrated_predictions,
    average="weighted",
    zero_division=0
)


calibrated_recall = recall_score(
    y_test,
    calibrated_predictions,
    average="weighted",
    zero_division=0
)


calibrated_f1 = f1_score(
    y_test,
    calibrated_predictions,
    average="weighted",
    zero_division=0
)


print("\nCALIBRATED MODEL PERFORMANCE")

print(
    f"Accuracy : {calibrated_accuracy:.4f}"
)

print(
    f"Precision: {calibrated_precision:.4f}"
)

print(
    f"Recall   : {calibrated_recall:.4f}"
)

print(
    f"F1 Score : {calibrated_f1:.4f}"
)


# ============================================================
# 15. SAVE FINAL MODEL
# ============================================================

MODEL_PATH = "models/best_model.pkl"

joblib.dump(
    calibrated_model,
    MODEL_PATH
)


print("\n" + "=" * 70)
print("FINAL MODEL")
print("=" * 70)

print(
    f"Final model: Calibrated {best_model_name}"
)

print(
    f"Model saved to: {MODEL_PATH}"
)


# ============================================================
# 16. SAVE MODEL RESULTS
# ============================================================

final_results = results_df.copy()


# Add calibrated model as a separate row
final_results.loc[
    "Calibrated " + best_model_name
] = [
    calibrated_accuracy,
    calibrated_precision,
    calibrated_recall,
    calibrated_f1
]


final_results = final_results.sort_values(
    by="F1 Score",
    ascending=False
)


final_results.to_csv(
    "outputs/model_results.csv"
)


print("\nModel comparison saved to:")
print("outputs/model_results.csv")


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    calibrated_predictions,
    labels=["FAIL", "PASS"]
)


plt.figure(
    figsize=(7, 5)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["FAIL", "PASS"],
    yticklabels=["FAIL", "PASS"]
)


plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(
    "Confusion Matrix - Calibrated Model"
)

plt.tight_layout()


plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)


plt.close()


print("Confusion matrix saved to:")
print("outputs/confusion_matrix.png")


# ============================================================
# 18. FEATURE IMPORTANCE / COEFFICIENTS
# ============================================================

try:

    # Get the original best pipeline
    fitted_preprocessor = (
        best_uncalibrated_model
        .named_steps["preprocessor"]
    )

    fitted_model = (
        best_uncalibrated_model
        .named_steps["model"]
    )

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )


    if hasattr(
        fitted_model,
        "coef_"
    ):

        importance_values = np.abs(
            fitted_model.coef_[0]
        )

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_values
        })


    elif hasattr(
        fitted_model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": (
                fitted_model
                .feature_importances_
            )
        })


    else:

        importance_df = None


    if importance_df is not None:

        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(15)
        )


        plt.figure(
            figsize=(10, 7)
        )


        plt.barh(
            importance_df["Feature"][::-1],
            importance_df["Importance"][::-1]
        )


        plt.xlabel("Importance")
        plt.ylabel("Feature")

        plt.title(
            "Important Features - Best Model"
        )

        plt.tight_layout()


        plt.savefig(
            "outputs/feature_importance.png",
            dpi=300
        )


        plt.close()


        print(
            "Feature importance saved to:"
        )

        print(
            "outputs/feature_importance.png"
        )


except Exception as e:

    print(
        "\nFeature importance could not be generated."
    )

    print(
        "Reason:",
        e
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"Dataset records       : {len(df)}"
)

print(
    f"Training records      : {len(X_train)}"
)

print(
    f"Testing records       : {len(X_test)}"
)

print(
    f"Best model            : "
    f"Calibrated {best_model_name}"
)

print(
    f"Calibrated accuracy   : "
    f"{calibrated_accuracy:.4f}"
)

print(
    f"Calibrated precision  : "
    f"{calibrated_precision:.4f}"
)

print(
    f"Calibrated recall     : "
    f"{calibrated_recall:.4f}"
)

print(
    f"Calibrated F1 score   : "
    f"{calibrated_f1:.4f}"
)


print("\nSaved files:")

print(
    "models/best_model.pkl"
)

print(
    "outputs/model_results.csv"
)

print(
    "outputs/confusion_matrix.png"
)

print(
    "outputs/feature_importance.png"
)


print(
    "\nThe model is ready for Streamlit."
)