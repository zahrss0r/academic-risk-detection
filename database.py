# database.py
#
# Where saved predictions are kept.
#
# One SQLite file, predictions.db, sitting next to the code. SQLite needs
# no server and no installation: it is one file, and sqlite3 comes with
# Python, so the project gains a database without gaining a dependency.
#
# WHAT IS STORED, AND WHAT IS NOT.
#
# Stored: the university number, the course code, the date, and the result
# of the prediction. That is what an advisor needs to see whether a
# student's situation is getting better or worse between two meetings.
#
# Not stored, ever: names, national identity numbers, phone numbers, email
# addresses, or anything else that identifies a person outside the
# university's own records. The marks the advisor types in are not stored
# either. Only the outcome of the calculation is.
#
# The Senior Project I report said the system would store nothing at all.
# That has changed, deliberately, and the privacy requirement in Chapter 4
# is rewritten to describe exactly the two tables below.

import sqlite3
import pandas as pd

import settings

# Beside the code, wherever the program was started from.
DATABASE_FILE = settings.project_file("predictions.db")


def connect():
    # Opens the database file, creating it the first time.
    return sqlite3.connect(DATABASE_FILE)


def create_tables():
    # Called once when the application starts. "IF NOT EXISTS" means
    # running it again on an existing file changes nothing.
    connection = connect()
    cursor = connection.cursor()

    # One row per student. The things that do not change during a term.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student (
            student_id        TEXT PRIMARY KEY,
            program           TEXT,
            level             INTEGER,
            credits_this_term INTEGER
        )
    """)

    # Many rows per student, one for every prediction that was saved. This
    # is the one to many relationship in the entity relationship diagram.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction (
            prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id       TEXT NOT NULL,
            course_code      TEXT NOT NULL,
            made_on          TEXT NOT NULL,
            day_of_term      INTEGER,
            mark_so_far      REAL,
            highest_possible REAL,
            probability      REAL,
            risk_level       TEXT,
            policy_status    TEXT,
            FOREIGN KEY (student_id) REFERENCES student (student_id)
        )
    """)

    connection.commit()
    connection.close()


def save_student(student_id, program, level, credits_this_term):
    # Adds the student if new, updates the details if already there.
    # "INSERT OR REPLACE" does both, because student_id is the primary key.
    connection = connect()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO student
            (student_id, program, level, credits_this_term)
        VALUES (?, ?, ?, ?)
    """, (student_id, program, level, credits_this_term))

    connection.commit()
    connection.close()


def save_prediction(student_id, course_code, made_on, day_of_term,
                    mark_so_far, highest_possible, probability,
                    risk_level, policy_status):
    # Adds one prediction. prediction_id is filled in by the database.
    #
    # The values are passed as question marks rather than glued into the
    # text of the query. That is the correct way to do it: a student number
    # typed into a box can never be read as part of the command.
    connection = connect()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO prediction
            (student_id, course_code, made_on, day_of_term, mark_so_far,
             highest_possible, probability, risk_level, policy_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (student_id, course_code, made_on, day_of_term, mark_so_far,
          highest_possible, probability, risk_level, policy_status))

    connection.commit()
    connection.close()


def get_history(student_id):
    # Every prediction saved for one student, newest first.
    connection = connect()

    history = pd.read_sql_query("""
        SELECT made_on, course_code, day_of_term, mark_so_far,
               highest_possible, probability, risk_level, policy_status
        FROM prediction
        WHERE student_id = ?
        ORDER BY prediction_id DESC
    """, connection, params=(student_id,))

    connection.close()
    return history


def count_predictions():
    # How many predictions are saved in total.
    connection = connect()
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM prediction")
    total = cursor.fetchone()[0]
    connection.close()
    return total


def count_students():
    connection = connect()
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM student")
    total = cursor.fetchone()[0]
    connection.close()
    return total


def export_tables():
    # Writes both tables out as plain CSV next to the database file.
    #
    # predictions.db is a real database file, which is what the submission
    # asks for, but it is binary: double clicking it opens nothing useful.
    # These two CSV files hold exactly the same rows in a form anybody can
    # open in Excel without installing a thing, so the contents can be
    # read without taking our word for what is in there.
    connection = connect()

    students = pd.read_sql_query("SELECT * FROM student", connection)
    predictions = pd.read_sql_query("SELECT * FROM prediction", connection)

    connection.close()

    students.to_csv(settings.project_file("table_student.csv"), index=False)
    predictions.to_csv(settings.project_file("table_prediction.csv"), index=False)

    return len(students), len(predictions)
