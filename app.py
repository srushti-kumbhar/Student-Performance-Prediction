import os
import sqlite3
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/student_performance.csv"
DATABASE_PATH = "database/predictions.db"
RESULT_PATH = "outputs/model_results.csv"

# Friendly labels for visible columns only.
LABELS = {
    "student_id": "Student ID",
    "gender": "Gender",
    "age": "Age",
    "class_year": "Class / Year",
    "semester": "Semester",
    "number_of_subjects": "Number of Subjects",
    "attendance": "Attendance (%)",
    "study_hours": "Study Hours / Day",
    "previous_percentage": "Previous Percentage",
    "previous_exam_score": "Previous Exam Score",
    "internal_marks": "Internal Marks",
    "assignment_marks": "Assignment Marks",
    "practical_marks": "Practical Marks",
    "project_presentation_marks": "Project / Presentation Marks",
    "quiz_test_scores": "Quiz / Test Scores",
    "study_consistency": "Study Consistency",
    "assignment_submission_rate": "Assignment Submission Rate (%)",
    "learning_study_method": "Learning / Study Method",
    "final_exam_marks": "Final Exam Marks",
    "backlogs": "Number of Backlogs",
    "extracurricular": "Extracurricular Activities",
    "prediction": "Predicted Result",
    "probability": "PASS Probability (%)",
    "performance_category": "Performance Category",
    "risk_level": "Risk Level",
    "timestamp": "Prediction Date / Time",
    "student_status": "Student Status",
    "updated_at": "Last Updated",
    "F1_Score": "F1 Score",
    "F1 Score": "F1 Score",
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "Model": "Model"
}

# These fields will not be displayed in Student Records tables.
# They remain available internally for model predictions.
HIDDEN_DISPLAY_COLUMNS = {
    "previous_exam_score",
    "quiz_test_scores",
    "assignment_submission_rate",
    "project_presentation_marks"
}


def friendly(df):
    """Hide selected columns and rename the remaining display labels."""
    visible = df.drop(
        columns=[
            col for col in HIDDEN_DISPLAY_COLUMNS
            if col in df.columns
        ],
        errors="ignore"
    )

    return visible.rename(
        columns={
            col: LABELS.get(col, col.replace("_", " ").title())
            for col in visible.columns
        }
    )


def model_results():
    if not os.path.exists(RESULT_PATH):
        return pd.DataFrame()

    df = pd.read_csv(RESULT_PATH)

    if "Model" not in df.columns and not df.empty:
        df = df.rename(columns={df.columns[0]: "Model"})

    if "F1_Score" in df.columns and "F1 Score" not in df.columns:
        df = df.rename(columns={"F1_Score": "F1 Score"})

    return df


# --------------------------- Database ---------------------------

def create_database():
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            gender TEXT,
            age INTEGER,
            class_year TEXT,
            semester INTEGER,
            number_of_subjects INTEGER,
            attendance REAL,
            study_hours REAL,
            previous_percentage REAL,
            previous_exam_score REAL,
            internal_marks REAL,
            assignment_marks REAL,
            practical_marks REAL,
            project_presentation_marks REAL,
            quiz_test_scores REAL,
            study_consistency TEXT,
            assignment_submission_rate REAL,
            learning_study_method TEXT,
            final_exam_marks REAL,
            backlogs INTEGER,
            extracurricular TEXT,
            prediction TEXT,
            probability REAL,
            performance_category TEXT,
            risk_level TEXT,
            timestamp TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            gender TEXT,
            age INTEGER,
            class_year TEXT,
            semester INTEGER,
            number_of_subjects INTEGER,
            attendance REAL,
            study_hours REAL,
            previous_percentage REAL,
            previous_exam_score REAL,
            internal_marks REAL,
            assignment_marks REAL,
            practical_marks REAL,
            project_presentation_marks REAL,
            quiz_test_scores REAL,
            study_consistency TEXT,
            assignment_submission_rate REAL,
            learning_study_method TEXT,
            final_exam_marks REAL,
            backlogs INTEGER,
            extracurricular TEXT,
            student_status TEXT DEFAULT 'Active',
            updated_at TEXT
        )
    """)

    additions = {
        "class_year": "TEXT",
        "semester": "INTEGER",
        "number_of_subjects": "INTEGER",
        "previous_exam_score": "REAL",
        "project_presentation_marks": "REAL",
        "quiz_test_scores": "REAL",
        "study_consistency": "TEXT",
        "assignment_submission_rate": "REAL",
        "learning_study_method": "TEXT",
        "final_exam_marks": "REAL"
    }

    for table in ("students", "predictions"):
        existing = {
            row[1]
            for row in cur.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }

        for column, data_type in additions.items():
            if column not in existing:
                cur.execute(
                    f"ALTER TABLE {table} ADD COLUMN "
                    f"{column} {data_type}"
                )

    conn.commit()
    conn.close()


PROFILE_COLS = [
    "student_id",
    "gender",
    "age",
    "class_year",
    "semester",
    "number_of_subjects",
    "attendance",
    "study_hours",
    "previous_percentage",
    "previous_exam_score",
    "internal_marks",
    "assignment_marks",
    "practical_marks",
    "project_presentation_marks",
    "quiz_test_scores",
    "study_consistency",
    "assignment_submission_rate",
    "learning_study_method",
    "final_exam_marks",
    "backlogs",
    "extracurricular"
]


def get_profile(sid, table):
    if not sid:
        return None

    cols = PROFILE_COLS.copy()

    if table == "students":
        cols.append("student_status")

    conn = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql_query(
        f"SELECT {','.join(cols)} FROM {table} "
        "WHERE student_id=? ORDER BY rowid DESC LIMIT 1",
        conn,
        params=(sid,)
    )

    conn.close()

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def save_student(values):
    cols = PROFILE_COLS + ["student_status", "updated_at"]

    vals = [values.get(col) for col in cols]
    vals[-1] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    updates = ", ".join(
        f"{col}=excluded.{col}"
        for col in cols
        if col not in ("student_id", "updated_at")
    )

    conn = sqlite3.connect(DATABASE_PATH)

    conn.execute(
        f"INSERT INTO students ({','.join(cols)}) "
        f"VALUES ({','.join(['?'] * len(cols))}) "
        f"ON CONFLICT(student_id) DO UPDATE SET "
        f"{updates}, updated_at=excluded.updated_at",
        vals
    )

    conn.commit()
    conn.close()


def save_prediction(values):
    cols = PROFILE_COLS + [
        "prediction",
        "probability",
        "performance_category",
        "risk_level",
        "timestamp"
    ]

    vals = [values.get(col) for col in cols]
    vals[-1] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DATABASE_PATH)

    conn.execute(
        f"INSERT INTO predictions ({','.join(cols)}) "
        f"VALUES ({','.join(['?'] * len(cols))})",
        vals
    )

    conn.commit()
    conn.close()


def read_table(table, extra):
    conn = sqlite3.connect(DATABASE_PATH)

    cols = PROFILE_COLS + extra

    df = pd.read_sql_query(
        f"SELECT {','.join(cols)} FROM {table} ORDER BY rowid DESC",
        conn
    )

    conn.close()
    return df


create_database()


# --------------------------- Load Model and Dataset ---------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run train_model.py first.")
        st.stop()

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found. Run generate_dataset.py first.")
        st.stop()

    return pd.read_csv(DATA_PATH)


model = load_model()
dataset = load_data()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎯 Prediction",
        "📋 Student Records",
        "📊 Model Performance",
        "ℹ️ About"
    ]
)


# --------------------------- Home ---------------------------

if page == "🏠 Home":
    st.title("🎓 Student Performance Prediction System")

    st.write(
        "A machine-learning-based educational prototype "
        "for predicting student PASS/FAIL performance."
    )

    a, b, c, d = st.columns(4)

    a.metric("Dataset Records", f"{len(dataset):,}")
    b.metric("Features", "20")
    c.metric("Age Range", "18–25")
    d.metric("Prediction Type", "PASS / FAIL")

    res = model_results()

    if not res.empty:
        best = res.iloc[0]
        x, y, z = st.columns(3)

        x.metric("Best Model", str(best.get("Model", "Unknown")))

        if "Accuracy" in res.columns:
            y.metric("Accuracy", f"{best['Accuracy'] * 100:.2f}%")

        if "F1 Score" in res.columns:
            z.metric("F1 Score", f"{best['F1 Score'] * 100:.2f}%")

    st.subheader("System Features")

    features = [
        "Machine-learning prediction",
        "PASS / FAIL probability",
        "Performance category and risk level",
        "Academic factors and recommendations",
        "Persistent student profiles",
        "SQLite prediction history"
    ]

    for item in features:
        st.write("✔ " + item)

    st.info(
        "The dataset is synthetic and does not represent "
        "actual Mumbai University data."
    )


# --------------------------- Prediction ---------------------------

elif page == "🎯 Prediction":
    st.title("🎯 Student Performance Prediction")

    st.info(
        "Enter a Student ID. Previously saved information "
        "is loaded when the ID already exists."
    )

    sid = st.text_input(
        "Student ID",
        value="STU00001"
    ).strip().upper()

    profile = get_profile(sid, "students") if sid else None

    if profile is None and sid:
        profile = get_profile(sid, "predictions")

    if profile:
        st.success(
            f"Existing student found: {sid}. Saved information loaded."
        )

    def val(key, default):
        value = profile.get(key) if profile else None

        if value is None or pd.isna(value):
            return default

        return value

    defaults = {
        "gender": val("gender", "Male"),
        "age": int(val("age", 20)),
        "class_year": val("class_year", "Third Year"),
        "semester": int(val("semester", 5)),
        "number_of_subjects": int(val("number_of_subjects", 6)),
        "attendance": float(val("attendance", 75)),
        "study_hours": float(val("study_hours", 5)),
        "previous_percentage": float(val("previous_percentage", 65)),
        "previous_exam_score": float(
            val("previous_exam_score", val("previous_percentage", 65))
        ),
        "internal_marks": float(val("internal_marks", 32)),
        "assignment_marks": float(val("assignment_marks", 14)),
        "practical_marks": float(val("practical_marks", 14)),
        "project_presentation_marks": float(
            val("project_presentation_marks", 14)
        ),
        "quiz_test_scores": float(val("quiz_test_scores", 14)),
        "study_consistency": val("study_consistency", "Regular"),
        "assignment_submission_rate": float(
            val("assignment_submission_rate", 75)
        ),
        "learning_study_method": val("learning_study_method", "Self Study"),
        "final_exam_marks": float(val("final_exam_marks", 65)),
        "backlogs": int(val("backlogs", 0)),
        "extracurricular": val("extracurricular", "No"),
        "student_status": val("student_status", "Active")
    }

    c1, c2, c3 = st.columns(3)

    with c1:
        genders = ["Male", "Female", "Other"]

        gender = st.selectbox(
            "Gender",
            genders,
            index=genders.index(defaults["gender"])
            if defaults["gender"] in genders else 0
        )

        age = st.number_input(
            "Age",
            18,
            25,
            max(18, min(25, defaults["age"])),
            1
        )

        years = [
            "First Year",
            "Second Year",
            "Third Year",
            "Fourth Year"
        ]

        cy = st.selectbox(
            "Class / Year",
            years,
            index=years.index(defaults["class_year"])
            if defaults["class_year"] in years else 2
        )

        semester = st.number_input(
            "Semester",
            1,
            8,
            max(1, min(8, defaults["semester"])),
            1
        )

        attendance = st.number_input(
            "Attendance (%)",
            0.0,
            100.0,
            max(0.0, min(100.0, defaults["attendance"])),
            0.5
        )

        study_hours = st.number_input(
            "Study Hours / Day",
            0.0,
            10.0,
            max(0.0, min(10.0, defaults["study_hours"])),
            0.5
        )

    with c2:
        previous_percentage = st.number_input(
            "Previous Percentage",
            0.0,
            100.0,
            max(0.0, min(100.0, defaults["previous_percentage"])),
            0.5
        )

        internal_marks = st.number_input(
            "Internal Marks / 50",
            0.0,
            50.0,
            max(0.0, min(50.0, defaults["internal_marks"])),
            0.5
        )

        assignment_marks = st.number_input(
            "Assignment Marks / 20",
            0.0,
            20.0,
            max(0.0, min(20.0, defaults["assignment_marks"])),
            0.5
        )

        practical_marks = st.number_input(
            "Practical Marks / 20",
            0.0,
            20.0,
            max(0.0, min(20.0, defaults["practical_marks"])),
            0.5
        )

        cons = st.selectbox(
            "Study Consistency",
            ["Regular", "Irregular"],
            index=["Regular", "Irregular"].index(
                defaults["study_consistency"]
            ) if defaults["study_consistency"] in ["Regular", "Irregular"]
            else 0
        )

        methods = [
            "Self Study",
            "Coaching",
            "Online Learning",
            "Mixed"
        ]

        method = st.selectbox(
            "Learning / Study Method",
            methods,
            index=methods.index(defaults["learning_study_method"])
            if defaults["learning_study_method"] in methods else 0
        )

    with c3:
        final_exam_marks = st.number_input(
            "Final Exam Marks",
            0.0,
            100.0,
            max(0.0, min(100.0, defaults["final_exam_marks"])),
            0.5
        )

        backlogs = st.number_input(
            "Number of Backlogs",
            0,
            10,
            max(0, defaults["backlogs"]),
            1
        )

        extra = st.selectbox(
            "Extracurricular Activities",
            ["Yes", "No"],
            index=["Yes", "No"].index(defaults["extracurricular"])
            if defaults["extracurricular"] in ["Yes", "No"] else 1
        )

        status = st.selectbox(
            "Student Status",
            ["Active", "Dropped"],
            index=["Active", "Dropped"].index(defaults["student_status"])
            if defaults["student_status"] in ["Active", "Dropped"] else 0
        )

    # Hidden features remain available to the trained model.
    subjects = defaults["number_of_subjects"]
    previous_exam_score = defaults["previous_exam_score"]
    project_presentation_marks = defaults["project_presentation_marks"]
    quiz_test_scores = defaults["quiz_test_scores"]
    assignment_submission_rate = defaults["assignment_submission_rate"]

    if st.button("🔮 Predict Performance", type="primary"):
        if not sid:
            st.error("Please enter a Student ID.")
            st.stop()

        X = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "attendance": [attendance],
            "study_hours": [study_hours],
            "previous_percentage": [previous_percentage],
            "internal_marks": [internal_marks],
            "assignment_marks": [assignment_marks],
            "practical_marks": [practical_marks],
            "backlogs": [backlogs],
            "extracurricular": [extra],
            "class_year": [cy],
            "semester": [semester],
            "number_of_subjects": [subjects],
            "previous_exam_score": [previous_exam_score],
            "project_presentation_marks": [project_presentation_marks],
            "quiz_test_scores": [quiz_test_scores],
            "study_consistency": [cons],
            "assignment_submission_rate": [assignment_submission_rate],
            "learning_study_method": [method],
            "final_exam_marks": [final_exam_marks]
        })

        prediction = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)

        pass_prob = float(probs[classes.index("PASS")] * 100)
        fail_prob = float(probs[classes.index("FAIL")] * 100)

        score = (
            0.10 * attendance
            + 0.08 * study_hours * 10
            + 0.10 * previous_percentage
            + 0.10 * previous_exam_score
            + 0.08 * internal_marks / 50 * 100
            + 0.07 * assignment_marks / 20 * 100
            + 0.07 * practical_marks / 20 * 100
            + 0.08 * project_presentation_marks / 20 * 100
            + 0.07 * quiz_test_scores / 20 * 100
            + 0.08 * assignment_submission_rate
            + 0.08 * (100 if cons == "Regular" else 50)
            + 0.10 * final_exam_marks
            - 2 * backlogs
        )

        score = max(0, min(100, score))

        if score >= 80:
            category = "Excellent"
        elif score >= 65:
            category = "Good"
        elif score >= 50:
            category = "Average"
        else:
            category = "Poor"

        if fail_prob >= 60:
            risk = "HIGH"
        elif fail_prob >= 30:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        st.divider()
        st.subheader("Prediction Result")

        st.caption(
            "PASS/FAIL comes from the trained machine-learning model; "
            "no single input is an automatic PASS/FAIL rule."
        )

        a, b, c, d = st.columns(4)

        a.metric("Predicted Result", str(prediction))
        b.metric("PASS Probability", f"{pass_prob:.2f}%")
        c.metric("FAIL Probability", f"{fail_prob:.2f}%")
        d.metric("Risk Level", risk)

        e, f = st.columns(2)
        e.metric("Performance Category", category)
        f.metric("Academic Profile Score", f"{score:.1f}/100")

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            ["PASS", "FAIL"],
            [pass_prob, fail_prob]
        )

        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        ax.set_title("PASS / FAIL Prediction Probability")
        ax.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, [pass_prob, fail_prob]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 2, 97),
                f"{value:.2f}%",
                ha="center"
            )

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Factors to Consider")

        factors = []

        if final_exam_marks >= 75:
            factors.append("Final exam performance is strong.")
        elif final_exam_marks < 50:
            factors.append(
                "Final exam performance is weak and may reduce "
                "the academic profile."
            )
        else:
            factors.append("Final exam marks are in the middle range.")

        if attendance < 75:
            factors.append("Attendance is below 75%.")
        else:
            factors.append("Attendance is satisfactory.")

        if cons == "Irregular":
            factors.append("Study consistency is irregular.")
        else:
            factors.append("Study consistency is regular.")

        if backlogs:
            factors.append(f"Student has {backlogs} backlog(s).")
        else:
            factors.append("No backlogs.")

        if practical_marks < 10:
            factors.append("Practical performance could be improved.")

        if internal_marks < 25:
            factors.append("Internal marks could be improved.")

        for item in factors:
            st.write("• " + item)

        st.subheader("Recommendations")
        recommendations = []

        if final_exam_marks < 50:
            recommendations.append(
                "Focus on final-exam preparation, revision and timed practice."
            )

        if attendance < 75:
            recommendations.append(
                "Improve regular class attendance where possible."
            )

        if study_hours < 4 or cons == "Irregular":
            recommendations.append(
                "Build a consistent daily study routine."
            )

        if internal_marks < 25:
            recommendations.append(
                "Focus on internal tests, classwork and revision."
            )

        if practical_marks < 10:
            recommendations.append(
                "Practice laboratory and practical work regularly."
            )

        if backlogs > 0:
            recommendations.append(
                "Clear pending backlogs as early as possible."
            )

        if not recommendations:
            recommendations.append(
                "Maintain the current balanced study routine."
            )

        for item in recommendations:
            st.write("• " + item)

        saved = {
            "student_id": sid,
            "gender": gender,
            "age": age,
            "class_year": cy,
            "semester": semester,
            "number_of_subjects": subjects,
            "attendance": attendance,
            "study_hours": study_hours,
            "previous_percentage": previous_percentage,
            "previous_exam_score": previous_exam_score,
            "internal_marks": internal_marks,
            "assignment_marks": assignment_marks,
            "practical_marks": practical_marks,
            "project_presentation_marks": project_presentation_marks,
            "quiz_test_scores": quiz_test_scores,
            "study_consistency": cons,
            "assignment_submission_rate": assignment_submission_rate,
            "learning_study_method": method,
            "final_exam_marks": final_exam_marks,
            "backlogs": backlogs,
            "extracurricular": extra,
            "student_status": status,
            "prediction": prediction,
            "probability": pass_prob,
            "performance_category": category,
            "risk_level": risk
        }

        save_student(saved)
        save_prediction(saved)

        st.success(
            "Prediction saved. Student profile and prediction history "
            "have been updated."
        )


# --------------------------- Student Records ---------------------------

elif page == "📋 Student Records":
    st.title("📋 Student Records")

    t1, t2 = st.tabs(["Student Profiles", "Prediction History"])

    with t1:
        profiles = read_table(
            "students",
            ["student_status", "updated_at"]
        )

        if profiles.empty:
            st.info("No student profiles available yet.")
        else:
            # The four unwanted columns are removed before display.
            st.dataframe(
                friendly(profiles),
                use_container_width=True
            )

            a, b, c = st.columns(3)

            a.metric("Unique Students", len(profiles))
            b.metric(
                "Active Students",
                int((profiles["student_status"] == "Active").sum())
            )
            c.metric(
                "Dropped Students",
                int((profiles["student_status"] == "Dropped").sum())
            )

    with t2:
        records = read_table(
            "predictions",
            [
                "prediction",
                "probability",
                "performance_category",
                "risk_level",
                "timestamp"
            ]
        )

        if records.empty:
            st.info("No prediction records available yet.")
        else:
            # The four unwanted columns are removed before display.
            st.dataframe(
                friendly(records),
                use_container_width=True
            )

            a, b, c = st.columns(3)

            a.metric("Total Predictions", len(records))
            b.metric(
                "PASS",
                int((records["prediction"] == "PASS").sum())
            )
            c.metric(
                "FAIL",
                int((records["prediction"] == "FAIL").sum())
            )


# --------------------------- Model Performance ---------------------------

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    results = model_results()

    if results.empty:
        st.warning("Model results not found. Please run train_model.py.")
    else:
        st.dataframe(
            friendly(results),
            use_container_width=True
        )

        if "Accuracy" in results.columns:
            st.subheader("Accuracy Comparison")

            st.bar_chart(
                results.set_index("Model")["Accuracy"] * 100
            )

        best = results.iloc[0]
        parts = [f"Best Model: {best.get('Model', 'Unknown')}"]

        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            if col in results.columns:
                parts.append(
                    f"{LABELS.get(col, col)}: {best[col] * 100:.2f}%"
                )

        st.success(" | ".join(parts))

    if os.path.exists("outputs/confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image(
            "outputs/confusion_matrix.png",
            use_container_width=True
        )

    if os.path.exists("outputs/feature_importance.png"):
        st.subheader("Feature Importance")
        st.image(
            "outputs/feature_importance.png",
            use_container_width=True
        )

        st.caption(
            "Feature importance shows model contribution, not causation. "
            "The labels inside this saved image are generated in "
            "train_model.py."
        )


# --------------------------- About ---------------------------

elif page == "ℹ️ About":
    st.title("ℹ️ About")

    st.write(
        "This project is a synthetic educational prototype for "
        "predicting student PASS/FAIL performance using machine learning."
    )

    st.write(
        f"The current dataset contains {len(dataset):,} "
        "synthetically generated student records."
    )

    st.write(
        "The dataset is simulated and does not represent actual "
        "Mumbai University student data."
    )

    st.subheader("Machine Learning")
    st.write(
        "The system compares Logistic Regression, Decision Tree, "
        "Random Forest and KNN."
    )

    results = model_results()

    if not results.empty:
        best = results.iloc[0]

        st.write(
            f"Current best model: {best.get('Model', 'Unknown')}"
        )

        if "Accuracy" in results.columns:
            st.write(
                f"Measured accuracy: {best['Accuracy'] * 100:.2f}%"
            )

        if "F1 Score" in results.columns:
            st.write(
                f"Measured F1 Score: {best['F1 Score'] * 100:.2f}%"
            )

    st.subheader("Age Validation")
    st.write("Valid student age range: 18 to 25 years.")

    st.subheader("Student Data Persistence")
    st.write(
        "Each Student ID is saved as a persistent profile. "
        "Entering the same ID loads saved information, and every "
        "new prediction is added to prediction history."
    )

    st.subheader("Technology")
    st.write(
        "Python, Pandas, NumPy, Scikit-learn, Streamlit, "
        "Matplotlib and SQLite."
    )