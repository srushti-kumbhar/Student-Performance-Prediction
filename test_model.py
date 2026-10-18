import joblib
import pandas as pd


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)

print("\n==========================================")
print(" SAVED MODEL TEST")
print("==========================================")

print("\nModel loaded successfully!")
print("Model type:", type(model.named_steps["model"]).__name__)


# ============================================================
# 2. CREATE SAMPLE STUDENT
# ============================================================

sample_student = pd.DataFrame([{
    "gender": "Female",
    "age": 20,
    "attendance": 85,
    "study_hours": 6,
    "previous_percentage": 75,
    "internal_marks": 78,
    "assignment_marks": 80,
    "practical_marks": 76,
    "backlogs": 0,
    "extracurricular": "Yes"
}])


# ============================================================
# 3. MAKE PREDICTION
# ============================================================

prediction = model.predict(sample_student)[0]

print("\nSample student prediction:")
print("Prediction:", prediction)


# ============================================================
# 4. GET PROBABILITY
# ============================================================

probabilities = model.predict_proba(sample_student)[0]

classes = model.named_steps["model"].classes_

print("\nPrediction probabilities:")

for class_name, probability in zip(classes, probabilities):
    print(
        f"{class_name}: {probability * 100:.2f}%"
    )


# ============================================================
# 5. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print(" MODEL TEST COMPLETED SUCCESSFULLY!")
print("==========================================")