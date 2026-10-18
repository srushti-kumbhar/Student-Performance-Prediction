import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer
)
from sklearn.inspection import permutation_importance


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/student_performance.csv"
MODEL_PATH = "models/best_model.pkl"

df = pd.read_csv(DATA_PATH)

print("\n==========================================")
print(" MODEL EVALUATION")
print("==========================================")


# ============================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["student_id", "final_result"])
y = df["final_result"]


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTest records:", len(X_test))


# ============================================================
# 4. LOAD BEST MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("\nBest model loaded successfully.")
print("Model:", type(model.named_steps["model"]).__name__)


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="PASS",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="PASS",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="PASS",
    zero_division=0
)


print("\n==========================================")
print(" EVALUATION METRICS")
print("==========================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["FAIL", "PASS"]
)

print("\n==========================================")
print(" CONFUSION MATRIX")
print("==========================================")

print("\n                 Predicted")
print("                 FAIL    PASS")
print(f"Actual FAIL     {cm[0][0]:5d}   {cm[0][1]:5d}")
print(f"Actual PASS     {cm[1][0]:5d}   {cm[1][1]:5d}")


# ============================================================
# 8. SAVE CONFUSION MATRIX IMAGE
# ============================================================

os.makedirs("outputs", exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FAIL", "PASS"]
)

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title("Student Performance Prediction - Confusion Matrix")
plt.tight_layout()

confusion_path = "outputs/confusion_matrix.png"

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved to:")
print(confusion_path)


# ============================================================
# 9. CUSTOM F1 SCORER FOR PASS
# ============================================================

f1_pass_scorer = make_scorer(
    f1_score,
    pos_label="PASS"
)


# ============================================================
# 10. PERMUTATION FEATURE IMPORTANCE
# ============================================================

print("\n==========================================")
print(" CALCULATING FEATURE IMPORTANCE")
print("==========================================")

print("Please wait...")


perm_result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring=f1_pass_scorer
)


# ============================================================
# 11. CREATE FEATURE IMPORTANCE TABLE
# ============================================================

feature_names = X_test.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm_result.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 12. DISPLAY FEATURE IMPORTANCE
# ============================================================

print("\nFeature importance:")

print(
    importance_df.to_string(
        index=False,
        formatters={
            "Importance": "{:.6f}".format
        }
    )
)


# ============================================================
# 13. SAVE FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Student Performance Model")

plt.gca().invert_yaxis()

plt.tight_layout()

feature_path = "outputs/feature_importance.png"

plt.savefig(
    feature_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance graph saved to:")
print(feature_path)


# ============================================================
# 14. SAVE FEATURE IMPORTANCE CSV
# ============================================================

importance_csv_path = "outputs/feature_importance.csv"

importance_df.to_csv(
    importance_csv_path,
    index=False
)

print("\nFeature importance data saved to:")
print(importance_csv_path)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print(" MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("==========================================")

print("\nGenerated files:")
print("1. outputs/confusion_matrix.png")
print("2. outputs/feature_importance.png")
print("3. outputs/feature_importance.csv")

print("\n==========================================\n")