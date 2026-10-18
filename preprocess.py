import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------
# Load dataset
# --------------------------------
file_path = "data/student_performance.csv"

df = pd.read_csv(file_path)


# --------------------------------
# Separate features and target
# --------------------------------
X = df.drop(columns=["student_id", "final_result"])
y = df["final_result"]


# --------------------------------
# Identify column types
# --------------------------------
categorical_features = [
    "gender",
    "extracurricular"
]

numerical_features = [
    "age",
    "attendance",
    "study_hours",
    "previous_percentage",
    "internal_marks",
    "assignment_marks",
    "practical_marks",
    "backlogs"
]


# --------------------------------
# Create preprocessing pipeline
# --------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# --------------------------------
# Train-test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# Fit preprocessing ONLY on training data
# --------------------------------
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# --------------------------------
# Display information
# --------------------------------
print("\n========================================")
print("DATA PREPROCESSING")
print("========================================")

print(f"\nTotal records: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nOriginal feature count:", X.shape[1])

print(
    "Processed training feature count:",
    X_train_processed.shape[1]
)

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

print("\nPreprocessing completed successfully.")
print("========================================")