import os
import numpy as np
import pandas as pd

# ============================================================
# STUDENT PERFORMANCE DATASET GENERATOR
# ============================================================

NUM_STUDENTS = 8000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "student_performance.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def clip_round(values, minimum, maximum, decimals=1):
    return np.round(
        np.clip(values, minimum, maximum),
        decimals
    )


# ============================================================
# BASIC STUDENT INFORMATION
# ============================================================

student_id = [
    f"STU{i:05d}"
    for i in range(1, NUM_STUDENTS + 1)
]

gender = rng.choice(
    ["Male", "Female"],
    size=NUM_STUDENTS,
    p=[0.55, 0.45]
)

age = rng.integers(
    18,
    26,
    size=NUM_STUDENTS
)

# Four class years with approximately equal representation
class_year = rng.choice(
    ["First Year", "Second Year", "Third Year", "Fourth Year"],
    size=NUM_STUDENTS
)

semester = rng.integers(
    1,
    9,
    size=NUM_STUDENTS
)

number_of_subjects = rng.integers(
    4,
    9,
    size=NUM_STUDENTS
)


# ============================================================
# ACADEMIC / STUDY FEATURES
# ============================================================

# Attendance
attendance = clip_round(
    rng.normal(80, 10, NUM_STUDENTS),
    40,
    100
)

# Daily study hours
study_hours = clip_round(
    rng.normal(5.2, 1.7, NUM_STUDENTS),
    1,
    10
)

# Previous academic percentage
previous_percentage = clip_round(
    rng.normal(70, 11, NUM_STUDENTS),
    35,
    95
)


# ============================================================
# PREVIOUS EXAM SCORE
# ============================================================

previous_exam_score = clip_round(
    (
        0.75 * previous_percentage
        + 0.25 * rng.normal(70, 12, NUM_STUDENTS)
        + rng.normal(0, 2.5, NUM_STUDENTS)
    ),
    30,
    98
)


# ============================================================
# INTERNAL MARKS
# ============================================================

internal_marks = clip_round(
    (
        0.78 * (previous_percentage / 100) * 50
        + 0.10 * (attendance - 70)
        + rng.normal(0, 1.8, NUM_STUDENTS)
    ),
    10,
    50
)


# ============================================================
# ASSIGNMENT MARKS
# ============================================================

assignment_marks = clip_round(
    (
        0.78 * (previous_percentage / 100) * 20
        + 0.10 * (study_hours - 5)
        + rng.normal(0, 0.9, NUM_STUDENTS)
    ),
    4,
    20
)


# ============================================================
# PRACTICAL MARKS
# ============================================================

practical_marks = clip_round(
    (
        0.75 * (previous_percentage / 100) * 20
        + 0.08 * (study_hours - 5)
        + rng.normal(0, 0.9, NUM_STUDENTS)
    ),
    5,
    20
)


# ============================================================
# PROJECT / PRESENTATION MARKS
# ============================================================

project_presentation_marks = clip_round(
    (
        0.72 * (previous_percentage / 100) * 20
        + 0.10 * (study_hours - 5)
        + rng.normal(0, 1.0, NUM_STUDENTS)
    ),
    5,
    20
)


# ============================================================
# QUIZ / TEST SCORES
# ============================================================

quiz_test_scores = clip_round(
    (
        0.75 * (previous_percentage / 100) * 20
        + 0.12 * (study_hours - 5)
        + rng.normal(0, 1.0, NUM_STUDENTS)
    ),
    5,
    20
)


# ============================================================
# STUDY CONSISTENCY
# ============================================================

consistency_score = (
    0.50 * attendance
    + 4.5 * study_hours
    + rng.normal(0, 9, NUM_STUDENTS)
)

study_consistency = np.where(
    consistency_score >= 70,
    "Regular",
    np.where(
        consistency_score >= 55,
        "Occasional",
        "Irregular"
    )
)


# ============================================================
# ASSIGNMENT SUBMISSION RATE
# ============================================================

assignment_submission_rate = clip_round(
    (
        62
        + 0.55 * (attendance - 75)
        + 3.0 * (study_hours - 5)
        + rng.normal(0, 9, NUM_STUDENTS)
    ),
    20,
    100
)


# ============================================================
# LEARNING / STUDY METHOD
# ============================================================

learning_study_method = rng.choice(
    [
        "Self Study",
        "Online Resources",
        "Classroom Focused",
        "Group Study"
    ],
    size=NUM_STUDENTS,
    p=[0.35, 0.25, 0.25, 0.15]
)


# ============================================================
# FINAL EXAM MARKS
# ============================================================
# Final exam is intentionally the strongest individual
# performance indicator, but NOT the only factor.
#
# This allows cases such as:
# - weak attendance + strong final exam -> PASS possible
# - strong attendance + weak final exam -> FAIL possible
# ============================================================

final_exam_marks = clip_round(
    (
        0.55 * previous_exam_score
        + 0.15 * (study_hours * 10)
        + 0.10 * attendance
        + 0.10 * (internal_marks * 2)
        + 0.10 * (quiz_test_scores * 5)
        + rng.normal(0, 7.5, NUM_STUDENTS)
    ),
    20,
    100
)


# ============================================================
# BACKLOGS
# ============================================================

backlog_probability = np.clip(
    0.18
    + 0.0025 * (60 - previous_percentage)
    + 0.0020 * (70 - attendance),
    0.03,
    0.35
)

backlog_flag = rng.random(NUM_STUDENTS) < backlog_probability

backlogs = np.where(
    backlog_flag,
    rng.integers(1, 3, NUM_STUDENTS),
    0
)


# ============================================================
# EXTRACURRICULAR ACTIVITIES
# ============================================================

extracurricular = rng.choice(
    ["Yes", "No"],
    size=NUM_STUDENTS,
    p=[0.45, 0.55]
)


# ============================================================
# FINAL PERFORMANCE SCORE
# ============================================================
# The target is created from multiple academic factors.
#
# Final exam has the highest individual contribution.
# Other academic factors still influence the result.
# Random noise prevents a perfectly deterministic target.
# ============================================================

overall_score = (
    0.32 * final_exam_marks
    + 0.12 * previous_percentage
    + 0.08 * previous_exam_score
    + 0.10 * (internal_marks * 2)
    + 0.06 * (assignment_marks * 5)
    + 0.06 * (practical_marks * 5)
    + 0.06 * (study_hours * 10)
    + 0.05 * attendance
    + 0.04 * assignment_submission_rate
    + 0.04 * (project_presentation_marks * 5)
    + 0.04 * (quiz_test_scores * 5)
)

# Small influence from consistency
overall_score += np.where(
    study_consistency == "Regular",
    2.0,
    np.where(
        study_consistency == "Occasional",
        0.5,
        -2.0
    )
)

# Small influence from learning method
overall_score += np.where(
    learning_study_method == "Self Study",
    1.0,
    np.where(
        learning_study_method == "Online Resources",
        0.7,
        np.where(
            learning_study_method == "Classroom Focused",
            0.5,
            0.3
        )
    )
)

# Backlogs negatively affect performance
overall_score -= 3.0 * backlogs

# Extracurricular activities have only a very small effect
overall_score += np.where(
    extracurricular == "Yes",
    0.5,
    0
)

# Moderate random variation
overall_score += rng.normal(
    0,
    2.0,
    NUM_STUDENTS
)


# ============================================================
# PASS / FAIL THRESHOLD
# ============================================================
# Approximately 55% PASS and 45% FAIL.
# The threshold is calculated from the generated scores,
# rather than manually assigning individual students.
# ============================================================

pass_threshold = np.percentile(
    overall_score,
    45
)

final_result = np.where(
    overall_score >= pass_threshold,
    "PASS",
    "FAIL"
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "student_id": student_id,
    "gender": gender,
    "age": age,
    "class_year": class_year,
    "semester": semester,
    "number_of_subjects": number_of_subjects,
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_percentage": previous_percentage,
    "previous_exam_score": previous_exam_score,
    "internal_marks": internal_marks,
    "assignment_marks": assignment_marks,
    "practical_marks": practical_marks,
    "project_presentation_marks": project_presentation_marks,
    "quiz_test_scores": quiz_test_scores,
    "study_consistency": study_consistency,
    "assignment_submission_rate": assignment_submission_rate,
    "learning_study_method": learning_study_method,
    "final_exam_marks": final_exam_marks,
    "backlogs": backlogs,
    "extracurricular": extracurricular,
    "final_result": final_result
})


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

assert len(df) == NUM_STUDENTS

assert df["student_id"].is_unique

assert df["age"].between(18, 25).all()

assert df["attendance"].between(40, 100).all()

assert df["study_hours"].between(1, 10).all()

assert df["previous_percentage"].between(35, 95).all()

assert df["final_exam_marks"].between(20, 100).all()

assert df["backlogs"].min() >= 0


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("=" * 70)
print("STUDENT PERFORMANCE DATASET GENERATED")
print("=" * 70)

print(f"\nDataset saved to: {OUTPUT_FILE}")

print(f"\nNumber of records: {len(df)}")

print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

print(f"Duplicate rows: {df.duplicated().sum()}")

print("\n" + "=" * 70)
print("PASS / FAIL DISTRIBUTION")
print("=" * 70)

print(df["final_result"].value_counts())

print("\nPercentage:")
print(
    df["final_result"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n" + "=" * 70)
print("NUMERICAL RANGES")
print("=" * 70)

range_columns = [
    "age",
    "attendance",
    "study_hours",
    "previous_percentage",
    "previous_exam_score",
    "internal_marks",
    "assignment_marks",
    "practical_marks",
    "project_presentation_marks",
    "quiz_test_scores",
    "assignment_submission_rate",
    "final_exam_marks",
    "backlogs"
]

for column in range_columns:
    print(
        f"{column:35s}: "
        f"{df[column].min():.2f} - {df[column].max():.2f}"
    )


# ============================================================
# CONFLICTING STUDENT CASES
# ============================================================
# These are intentionally allowed by the dataset.
# They demonstrate that the prediction is not based on
# attendance alone.
# ============================================================

weak_regular_profile = (
    (df["attendance"] >= 85)
    & (df["assignment_submission_rate"] >= 85)
    & (df["final_exam_marks"] <= 45)
)

strong_exam_profile = (
    (df["attendance"] <= 65)
    & (df["assignment_submission_rate"] <= 60)
    & (df["final_exam_marks"] >= 75)
)

print("\n" + "=" * 70)
print("CONFLICTING PROFILE CHECK")
print("=" * 70)

print(
    "Good attendance/submission + weak final exam:",
    int(weak_regular_profile.sum())
)

print(
    "Weak attendance/submission + strong final exam:",
    int(strong_exam_profile.sum())
)

print("\n" + "=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

print(df.head())

print("\n" + "=" * 70)
print("DATASET GENERATION COMPLETED SUCCESSFULLY")
print("=" * 70)