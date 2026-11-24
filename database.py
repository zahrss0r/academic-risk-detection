# database.py
#
# Stores everything the system has scanned, in one SQLite file
# (predictions.db) beside the code.
#
# Five tables:
#   cohort_file  every uploaded file, with a fingerprint of its contents
#   scan_run     every scan of a file; one file can be scanned more than once
#   scan_result  every flagged student in a run, with the reason given
#   notice       the students sent to an advisor
#   student      the few details about a student that do not change in a term
#
# Stored: the university number, the course code, the date, the reason, and
# the uploaded file as it arrived.
#
# Never stored: names, national identity numbers, phone numbers, email
# addresses, or anything else identifying a person.
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
    # Called once at startup. "IF NOT EXISTS" makes a second run harmless.
    connection = connect()
    cursor = connection.cursor()

    # One row per student, holding only what a cohort file carries about them.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student (
            student_id TEXT PRIMARY KEY,
            first_seen TEXT
        )
    """)

    # Every uploaded file, kept whole. The fingerprint is taken from the
    # contents, so a renamed file still matches and an edited one does not.
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

    # One row per scan. A file can be scanned again, so this is separate
    # from the file's own row.
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
            indicators_carried INTEGER,
            FOREIGN KEY (file_id) REFERENCES cohort_file (file_id)
        )
    """)

    # Every flagged student in a run, with the reason stored as given, so an
    # old scan reads as it did even if the model changes.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_result (
            result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL,
            student_id      TEXT NOT NULL,
            course_code     TEXT NOT NULL,
            reason          TEXT,
            reason_en       TEXT,
            probability     REAL,
            indicators_read INTEGER,
            FOREIGN KEY (run_id) REFERENCES scan_run (run_id)
        )
    """)

    # Many rows per student, one per notice issued.
    #
    #   probability      what the model gave, between 0 and 1
    #   reason           the sentence shown to the advisor
    #   indicators_read  how many indicators carried a value
    #   state            'waiting' or 'sent'. There is no third state.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notice (
            notice_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id       TEXT NOT NULL,
            course_code      TEXT NOT NULL,
            made_on          TEXT NOT NULL,
            day_of_term      INTEGER,
            reason           TEXT,
            probability      REAL,
            indicators_read  INTEGER,
            state            TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES student (student_id)
        )
    """)

    connection.commit()
    connection.close()


def save_student(student_id, first_seen):
    # "INSERT OR IGNORE" keeps the first date for a student already stored.
    connection = connect()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO student (student_id, first_seen)
        VALUES (?, ?)
    """, (student_id, first_seen))

    connection.commit()
    connection.close()


def fingerprint_of(contents):
    """A short string taken from the contents of a file.

    One changed character gives a completely different result, so two
    fingerprints can be compared instead of two files.
    """
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def find_file(contents):
    """Has this exact file been uploaded before? Returns its row or None.

    Matched on the fingerprint, so a renamed file is still recognised.
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
    """Store an uploaded file whole and return its file_id. Identical
    contents are not written twice."""
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
    """Files stored under this name whose contents are different."""
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
             flagged, indicators_carried):
    """Records one scan and hands back its run_id."""
    connection = connect()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO scan_run
            (file_id, ran_on, day_of_term, threshold, students, records,
             flagged, indicators_carried)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (file_id, ran_on, day_of_term, threshold, students, records,
          flagged, indicators_carried))
    connection.commit()
    run_id = cursor.lastrowid
    connection.close()
    return run_id


def save_results(run_id, rows):
    """Store every flagged record of one run.

    rows is a list of dictionaries, written together in one connection.
    """
    connection = connect()
    cursor = connection.cursor()
    for row in rows:
        cursor.execute("""
            INSERT INTO scan_result
                (run_id, student_id, course_code, reason,
                 reason_en, probability, indicators_read)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, row["student_id"], row["course_code"],
              row["reason"], row.get("reason_en"), row["probability"],
              row["indicators_read"]))
    connection.commit()
    connection.close()


def get_runs():
    """Every scan ever run, newest first, with the name of its file."""
    connection = connect()
    runs = pd.read_sql_query("""
        SELECT r.run_id, r.file_id, f.file_name, r.ran_on, r.students,
               r.records, r.flagged,
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
        SELECT student_id, course_code, reason, reason_en,
               probability, indicators_read
        FROM scan_result
        WHERE run_id = ?
        ORDER BY probability DESC, student_id
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


def save_notice(student_id, course_code, made_on, day_of_term,
                reason, probability, indicators_read, state):
    # Adds one notice; notice_id is filled in by the database. Values are
    # passed as parameters, so a student number can never be read as part
    # of the command.
    connection = connect()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO notice
            (student_id, course_code, made_on, day_of_term,
             reason, probability, indicators_read, state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (student_id, course_code, made_on, day_of_term,
          reason, probability, indicators_read, state))

    connection.commit()
    connection.close()


def notices_already_sent():
    """Every student who has already been notified, and when.

    A notice is a record of something that actually happened, so the screen
    reads it back from this table rather than holding it in memory, where
    closing the browser would lose it.
    """
    connection = connect()
    rows = pd.read_sql_query("""
        SELECT student_id, MAX(made_on) AS made_on
        FROM notice
        WHERE state = 'sent'
        GROUP BY student_id
    """, connection)
    connection.close()

    sent = {}
    for i in range(len(rows)):
        sent[rows["student_id"].iloc[i]] = rows["made_on"].iloc[i]
    return sent
