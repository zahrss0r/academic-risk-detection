"""
settings.py
Smart Academic Risk Detection System

Every number and every list the rest of the system depends on, kept in
one place so a change is made once.
"""

import os


# ---------------------------------------------------------------
# Finding the project's own files
# ---------------------------------------------------------------

# The folder this file sits in. Every data file is beside it.
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def project_file(name):
    """The full path to a file beside the code.

    A bare file name would fail when the program is started from another
    folder, as it is on Streamlit Cloud.
    """
    return os.path.join(PROJECT_FOLDER, name)


# ---------------------------------------------------------------
# The model
# ---------------------------------------------------------------

# Fixed so every run of every script gives the same answer.
RANDOM_SEED = 42

# The prediction moment: day 110 of the course run. Chosen from the
# courses themselves; the test and the window it leaves are in section 3.4
# of the report. Nothing dated after this day is used anywhere.
CUT_DAY = 110

# Raise an alert above this probability.
#
# Not 0.50. Missing a student costs that student a term; a false alarm
# costs a phone call. At 0.20 on the test slice: recall 0.8984, precision
# 0.6159, 91 missed of 896, 502 false alarms (results_final.csv). The
# full table of eleven thresholds is in results_threshold_table.csv.
#
# A low threshold is also the fairer one. On the validation slice the
# widest recall gap by region is 0.1605 at 0.20 (results_fairness.csv).
ALERT_THRESHOLD = 0.20


def flags(probability):
    """Whether a probability is reported. Works on one value or a column."""
    # The threshold belongs to the band above it, so a probability exactly
    # at it is reported. Written once here and used wherever a student is
    # flagged, so the rule cannot differ between the system and its tests.
    return probability >= ALERT_THRESHOLD


# The six indicators the model reads, in the order it expects them.
#
# Selected from seventy-one candidates by forward selection, then
# re-tested on the validation slice; the full path and the removal test
# are in results_10_full_path.csv and results_indicator_check.csv.
#
# No column about who the student is was kept. highest_education is
# academic background, not a personal characteristic.
INDICATORS = [
    "score_mean",         # the mean of the marks so far
    "weight_sum",         # the weight of the work handed in
    "visit_day_last",     # the day of the last visit to the platform
    "late_days_mean",     # how late, on average, the work was
    "clicks_homepage",    # clicks on the course home page
    "highest_education",  # the highest qualification held
]

# Kept in the processed files so the model can be audited, and never
# given to it. At 0.20 on the validation slice the widest recall gap is
# 0.0286 by gender and 0.0198 by age band, but 0.1605 by region and
# 0.0958 by deprivation band (results_fairness.csv). The model reads
# none of these columns: the unevenness is in the study behaviour.
AUDIT_ONLY_COLUMNS = [
    "gender",
    "region",
    "age_band",
    "imd_band",
    "disability",
    "final_result",
]

# The kinds of event row a cohort file can carry.
EVENT_ENROLMENT = "enrolment"     # one row per student per course
EVENT_SUBMISSION = "submission"   # one piece of work handed in
EVENT_ACTIVITY = "activity"       # one day's clicks on one page

# An enrolment row is required. A student who handed in nothing and never
# opened the platform has no submission row and no activity row, so
# without it the system would not know they were registered. In the
# training data 191 of 9,761 records are such students, and 99.0 per
# cent of them did not finish.

# The columns an event file is read for. A missing one is not a refusal:
# the indicator built from it is left empty.
#
# event_type is the exception. It says what each row is, so without it
# every indicator would be empty and the scan stops.
EVENT_COLUMNS = [
    "event_type",       # submission or activity
    "item_id",          # which piece of work, or which page
    "date_due",         # when the work was due        (submission)
    "date_submitted",   # when it was handed in        (submission)
    "score",            # the mark, out of 100         (submission)
    "weight",           # its weight in the course     (submission)
    "activity_type",    # what kind of page            (activity)
    "date",             # the day of the visit         (activity)
    "clicks",           # how many clicks that day     (activity)
]

# The event columns that have to read as numbers. Text in one of these is
# read as not known rather than as a zero, and the reader reports it.
NUMBER_COLUMNS = [
    "date_due",
    "date_submitted",
    "score",
    "weight",
    "date",
    "clicks",
]

# The three columns that say whose record a row is.
KEY_COLUMNS = [
    "code_module",
    "code_presentation",
    "id_student",
]

# ---------------------------------------------------------------
# Adding an indicator
# ---------------------------------------------------------------
#
# A university whose records carry a measurement this data does not have -
# a cumulative grade point average, say, or an attendance rate - cannot get
# a prediction out of it by adding the column to INDICATORS alone. A model
# reads only what it was trained on, and handing a trained model a number it
# has never seen gives noise rather than a prediction.
#
# The steps, in order:
#   1. add the name to INDICATORS above
#   2. make sure process_data.py builds a column with that name
#   3. run process_data.py, then select_indicators_full.py from the
#      experiment folder, then check_indicators.py, because a new candidate
#      can change which of the others still earn their place
#   4. run train_models.py to save the model the system will use
