# database.py
#
# Where everything the system has scanned is kept.
#
# One SQLite file, predictions.db, sitting next to the code. SQLite needs
# no server and no installation: it is one file, and sqlite3 comes with
# Python, so the project gains a database without gaining a dependency.
#
# FIVE TABLES, AND WHY EACH ONE EXISTS
#
#   cohort_file  every file that was ever uploaded, contents and all, with
#                a fingerprint of those contents so the same file can be
#                recognised if it arrives again
#   scan_run     every time a file was scanned. One file can be scanned
#                more than once, so this is separate from the file itself
#   scan_result  every student the system flagged in a run, with the reason
#                it gave. This is what lets an old scan be opened again and
#                read exactly as it was, without running it a second time
#   notice       the students the committee actually sent to an advisor
#   student      the few things about a student that do not change in a term
#
# WHAT IS STORED, AND WHAT IS NOT.
#
# Stored: the university number, the course code, the date, the reason the
# system gave, and the uploaded file as it arrived.
#
# Not stored, ever: names, national identity numbers, phone numbers, email
# addresses, or anything else that identifies a person outside the
# university's own records.
#
# The Senior Project I report said the system would store nothing at all.
# That has changed, deliberately, and the privacy requirement in Chapter 4
# is rewritten to describe exactly the tables below.

import hashlib
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

    # Many rows per student, one for every notice the system issued. This is
    # the one to many relationship in the entity relationship diagram.
    #
    # The columns follow the system exactly:
    #   caught_by        'rule' or 'model', never both, because a row a rule
    #                    caught is never put to the model
    #   probability      empty for a rule-caught row, for the same reason
    #   reason           the sentence the advisor was shown, computed from
    #                    that student's own row and stored as it was shown
    #   indicators_read  how many of the indicators carried a value, so a
    #                    notice can be read later knowing how much of the
    #                    picture the system actually had
    #   state            'waiting' or 'sent'. There is no third state: the
    #                    system notifies and stops, so what the advisor did
    #                    afterwards is not stored here.
    # Every file that was ever uploaded, kept whole.
    #
    # fingerprint is worked out from the contents, not the name. Two files
    # with the same name but one changed mark have different fingerprints,
    # and that is exactly the case we have to tell apart: the same file
    # arriving again is not a new cohort, but an updated file is.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cohort_file (
            file_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name    TEXT NOT NULL,
            fingerprint  TEXT NOT NULL,
            size_bytes   INTEGER,
            rows_in_file INTEGER,
            uploaded_on  TEXT NOT NULL,
            contents     TEXT NOT NULL
        )
    """)

    # One row every time a file is scanned. A file can be scanned again
    # later, which is why this is not part of the file's own row.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_run (
            run_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id            INTEGER NOT NULL,
            ran_on             TEXT NOT NULL,
            day_of_term        INTEGER,
            threshold          REAL,
            students           INTEGER,
            records            INTEGER,
            flagged            INTEGER,
            by_rule            INTEGER,
            by_model           INTEGER,
            indicators_carried INTEGER,
            FOREIGN KEY (file_id) REFERENCES cohort_file (file_id)
        )
    """)

    # Every student a run flagged, with the reason that run gave. Writing
    # the reason down rather than working it out again later means an old
    # scan reads exactly as it read on the day, even if the model changes.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_result (
            result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL,
            student_id      TEXT NOT NULL,
            course_code     TEXT NOT NULL,
            caught_by       TEXT NOT NULL,
            reason          TEXT,
            probability     REAL,
            indicators_read INTEGER,
            FOREIGN KEY (run_id) REFERENCES scan_run (run_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notice (
            notice_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id       TEXT NOT NULL,
            course_code      TEXT NOT NULL,
            made_on          TEXT NOT NULL,
            day_of_term      INTEGER,
            caught_by        TEXT NOT NULL,
            reason           TEXT,
            probability      REAL,
            indicators_read  INTEGER,
            state            TEXT NOT NULL,
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


def fingerprint_of(contents):
    """A short string worked out from the contents of a file.

    Change one character anywhere in the file and this comes out
    completely different, so comparing two fingerprints is a fair way of
    asking "is this the same file?" without comparing the files themselves
    character by character.
    """
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def find_file(contents):
    """Has this exact file been uploaded before? Returns its row or None.

    The lookup is on the fingerprint, so a file that was renamed is still
    recognised, and a file that kept its name but had a mark changed is
    correctly treated as a different file.
    """
    connection = connect()
    found = pd.read_sql_query("""
        SELECT file_id, file_name, uploaded_on, rows_in_file, size_bytes
        FROM cohort_file
        WHERE fingerprint = ?
    """, connection, params=(fingerprint_of(contents),))
    connection.close()

    if len(found) == 0:
        return None
    return found.iloc[0]


def save_file(file_name, contents, rows_in_file, uploaded_on):
    """Stores an uploaded file whole, and hands back its file_id.

    If the same contents are already here, nothing is written twice and the
    row that is already stored is the one returned.
    """
    already = find_file(contents)
    if already is not None:
        return int(already["file_id"])

    connection = connect()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO cohort_file
            (file_name, fingerprint, size_bytes, rows_in_file, uploaded_on,
             contents)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (file_name, fingerprint_of(contents), len(contents.encode("utf-8")),
          rows_in_file, uploaded_on, contents))
    connection.commit()
    file_id = cursor.lastrowid
    connection.close()
    return file_id


def other_versions(file_name, contents):
    """Files stored under this name whose contents are different.

    This is how the system can say "a file of this name is already here,
    but what you have just uploaded is not the same file."
    """
    connection = connect()
    found = pd.read_sql_query("""
        SELECT file_id, uploaded_on, rows_in_file
        FROM cohort_file
        WHERE file_name = ? AND fingerprint != ?
        ORDER BY file_id DESC
    """, connection, params=(file_name, fingerprint_of(contents)))
    connection.close()
    return found


def save_run(file_id, ran_on, day_of_term, threshold, students, records,
             flagged, by_rule, by_model, indicators_carried):
    """Records one scan and hands back its run_id."""
    connection = connect()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO scan_run
            (file_id, ran_on, day_of_term, threshold, students, records,
             flagged, by_rule, by_model, indicators_carried)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (file_id, ran_on, day_of_term, threshold, students, records,
          flagged, by_rule, by_model, indicators_carried))
    connection.commit()
    run_id = cursor.lastrowid
    connection.close()
    return run_id


def save_results(run_id, rows):
    """Stores every flagged record of one run.

    rows is a list of dictionaries, one per flagged course record. They go
    in together rather than one at a time because a scan produces a few
    hundred of them and opening the database once is enough.
    """
    connection = connect()
    cursor = connection.cursor()
    for row in rows:
        cursor.execute("""
            INSERT INTO scan_result
                (run_id, student_id, course_code, caught_by, reason,
                 probability, indicators_read)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, row["student_id"], row["course_code"], row["caught_by"],
              row["reason"], row["probability"], row["indicators_read"]))
    connection.commit()
    connection.close()


def get_runs():
    """Every scan ever run, newest first, with the name of its file."""
    connection = connect()
    runs = pd.read_sql_query("""
        SELECT r.run_id, r.file_id, f.file_name, r.ran_on, r.students,
               r.records, r.flagged, r.by_rule, r.by_model,
               r.indicators_carried, f.uploaded_on
        FROM scan_run r
        JOIN cohort_file f ON f.file_id = r.file_id
        ORDER BY r.run_id DESC
    """, connection)
    connection.close()
    return runs


def get_results(run_id):
    """Everything one run flagged, read back exactly as it was written."""
    connection = connect()
    rows = pd.read_sql_query("""
        SELECT student_id, course_code, caught_by, reason, probability,
               indicators_read
        FROM scan_result
        WHERE run_id = ?
        ORDER BY caught_by DESC, student_id
    """, connection, params=(run_id,))
    connection.close()
    return rows


def get_file_contents(file_id):
    """The stored file itself, so it can be scanned again."""
    connection = connect()
    cursor = connection.cursor()
    cursor.execute("SELECT contents FROM cohort_file WHERE file_id = ?",
                   (file_id,))
    row = cursor.fetchone()
    connection.close()
    if row is None:
        return None
    return row[0]


def save_notice(student_id, course_code, made_on, day_of_term, caught_by,
                reason, probability, indicators_read, state):
    # Adds one notice. notice_id is filled in by the database.
    #
    # The values are passed as question marks rather than glued into the
    # text of the query. That is the correct way to do it: a student number
    # read from a file can never be read as part of the command.
    connection = connect()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO notice
            (student_id, course_code, made_on, day_of_term, caught_by,
             reason, probability, indicators_read, state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (student_id, course_code, made_on, day_of_term, caught_by,
          reason, probability, indicators_read, state))

    connection.commit()
    connection.close()


def get_notices(student_id):
    # Every notice issued for one student, newest first.
    connection = connect()

    history = pd.read_sql_query("""
        SELECT made_on, course_code, day_of_term, caught_by, reason,
               probability, indicators_read, state
        FROM notice
        WHERE student_id = ?
        ORDER BY notice_id DESC
    """, connection, params=(student_id,))

    connection.close()
    return history


def count_notices():
    # How many notices are saved in total.
    connection = connect()
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM notice")
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
    # Writes the tables out as plain CSV next to the database file.
    #
    # predictions.db is a real database file, which is what the submission
    # asks for, but it is binary: double clicking it opens nothing useful.
    # These CSV files hold exactly the same rows in a form anybody can open
    # in Excel without installing a thing, so the contents can be read
    # without taking our word for what is in there.
    #
    # The uploaded files themselves are left out of this export on purpose.
    # They are whole cohort files sitting in one column, and writing them
    # into a CSV would produce something nobody can read. The list of which
    # files are stored goes out instead, without their contents.
    connection = connect()

    students = pd.read_sql_query("SELECT * FROM student", connection)
    notices = pd.read_sql_query("SELECT * FROM notice", connection)
    files = pd.read_sql_query("""
        SELECT file_id, file_name, fingerprint, size_bytes, rows_in_file,
               uploaded_on
        FROM cohort_file
    """, connection)
    runs = pd.read_sql_query("SELECT * FROM scan_run", connection)
    results = pd.read_sql_query("SELECT * FROM scan_result", connection)

    connection.close()

    students.to_csv(settings.project_file("table_student.csv"), index=False)
    notices.to_csv(settings.project_file("table_notice.csv"), index=False)
    files.to_csv(settings.project_file("table_cohort_file.csv"), index=False)
    runs.to_csv(settings.project_file("table_scan_run.csv"), index=False)
    results.to_csv(settings.project_file("table_scan_result.csv"), index=False)

    return len(students), len(notices)
