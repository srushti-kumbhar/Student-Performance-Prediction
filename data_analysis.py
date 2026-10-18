import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Load dataset
# -----------------------------
file_path = "data/student_performance.csv"

df = pd.read_csv(file_path)


# -----------------------------
# Create outputs folder
# -----------------------------
os.makedirs("outputs", exist_ok=True)


# -----------------------------
# Basic dataset information
# -----------------------------
print("\n========================================")
print("DATASET ANALYSIS")
print("========================================")

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# PASS/FAIL distribution
# -----------------------------
print("\nPASS/FAIL distribution:")
print(df["final_result"].value_counts())


# -----------------------------
# Visualization 1:
# PASS/FAIL distribution
# -----------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="final_result"
)

plt.title("Student PASS/FAIL Distribution")
plt.xlabel("Final Result")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig(
    "outputs/pass_fail_distribution.png",
    dpi=300
)

plt.close()


# -----------------------------
# Visualization 2:
# Attendance vs Final Result
# -----------------------------
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="final_result",
    y="attendance"
)

plt.title("Attendance vs Final Result")
plt.xlabel("Final Result")
plt.ylabel("Attendance (%)")
plt.tight_layout()

plt.savefig(
    "outputs/attendance_vs_result.png",
    dpi=300
)

plt.close()


# -----------------------------
# Visualization 3:
# Study Hours vs Final Result
# -----------------------------
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="final_result",
    y="study_hours"
)

plt.title("Study Hours vs Final Result")
plt.xlabel("Final Result")
plt.ylabel("Study Hours")
plt.tight_layout()

plt.savefig(
    "outputs/study_hours_vs_result.png",
    dpi=300
)

plt.close()


# -----------------------------
# Visualization 4:
# Previous Percentage vs Result
# -----------------------------
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="final_result",
    y="previous_percentage"
)

plt.title("Previous Percentage vs Final Result")
plt.xlabel("Final Result")
plt.ylabel("Previous Percentage")
plt.tight_layout()

plt.savefig(
    "outputs/previous_percentage_vs_result.png",
    dpi=300
)

plt.close()


# -----------------------------
# Visualization 5:
# Correlation heatmap
# -----------------------------
numeric_columns = [
    "age",
    "attendance",
    "study_hours",
    "previous_percentage",
    "internal_marks",
    "assignment_marks",
    "practical_marks",
    "backlogs"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300
)

plt.close()


# -----------------------------
# Final message
# -----------------------------
print("\nVisualizations created successfully:")

print("1. outputs/pass_fail_distribution.png")
print("2. outputs/attendance_vs_result.png")
print("3. outputs/study_hours_vs_result.png")
print("4. outputs/previous_percentage_vs_result.png")
print("5. outputs/correlation_heatmap.png")

print("\n========================================")
print("PHASE 3 ANALYSIS COMPLETED")
print("========================================")