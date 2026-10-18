import sqlite3


def create_database():
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            attendance REAL,
            study_hours REAL,
            previous_score REAL,
            assignment_score REAL,
            internal_score REAL,
            performance TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_student(name, attendance, study_hours, previous_score,
                 assignment_score, internal_score, performance):

    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO students
        (name, attendance, study_hours, previous_score,
         assignment_score, internal_score, performance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        attendance,
        study_hours,
        previous_score,
        assignment_score,
        internal_score,
        performance
    ))

    conn.commit()
    conn.close()


def get_students():
    conn = sqlite3.connect("students.db")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, attendance, study_hours,
               previous_score, assignment_score,
               internal_score, performance
        FROM students
    """)

    data = cursor.fetchall()

    conn.close()

    return data