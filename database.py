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
# The privacy requirement in Chapter 4 describes exactly the tables below.
#
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

    # One row per student the system has ever read, and only what a cohort
    # file actually carries about them. A table is either filled by the
    # system that owns it, or it does not belong in it.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student (
            student_id TEXT PRIMARY KEY,
            first_seen TEXT
        )
    """)

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
            reason          TEXT,
            reason_en       TEXT,
            probability     REAL,
            indicators_read INTEGER,
            FOREIGN KEY (run_id) REFERENCES scan_run (run_id)
        )
    """)

    # Many rows per student, one for every notice the system issued. This is
    # the one to many relationship in the entity relationship diagram.
    #
    # The columns follow the system exactly:
    #   probability      what the model gave this student, between 0 and 1
    #   reason           the sentence the advisor was shown, computed from
    #                    that student's own row and stored as it was shown
    #   indicators_read  how many of the indicators carried a value, so a
    #                    notice can be read later knowing how much of the
    #                    picture the system actually had
    #   state            'waiting' or 'sent'. There is no third state: the
    #                    system notifies and stops, so what the advisor did
    #                    afterwards is not stored here.
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
    # Adds the student if new and leaves the first date alone if not.
    # "INSERT OR IGNORE" does exactly that: a student already in the table
    # keeps the day we first saw them, however many files they appear in.
    connection = connect()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO student (student_id, first_seen)
        VALUES (?, ?)
    """, (student_id, first_seen))

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
    # Adds one notice. notice_id is filled in by the database.
    #
    # The values are passed as question marks rather than glued into the
    # text of the query. That is the correct way to do it: a student number
    # read from a file can never be read as part of the command.
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
