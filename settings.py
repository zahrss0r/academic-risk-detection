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

# The prediction moment: day 110 of the course run.
#
# Chosen from the courses themselves: at least two weighted pieces of
# work have fallen due and at least two are still ahead. That leaves a
# window of day 68 to 116 across the fourteen runs, and at day 110 about
# 43 per cent of the coursework weight is visible. Nothing dated after
# this day is used anywhere.
CUT_DAY = 110

# Raise an alert above this probability.
#
# Not 0.50. Missing a student costs that student a term; a false alarm
# costs a phone call. At 0.20 on the test slice: recall 0.8984, precision
# 0.6159, 91 missed of 896, 502 false alarms (results_final.csv). The
# full table of eleven thresholds is in results_threshold_table.csv.
#
# A low threshold is also the fairer one. On the validation slice the
# widest recall gap by region is 0.161 at 0.20 against 0.189 at 0.30.
ALERT_THRESHOLD = 0.20

# One threshold, and no bands above it.

# The six indicators the model reads, in the order it expects them.
#
# Chosen by a forward-selection experiment over seventy-one candidates,
# stopping at the last step that raised cross-validated ROC-AUC by 0.001
# or more. The path kept eight:
#       score_mean          0.8130   (+0.3130)
#       weight_sum          0.8581   (+0.0451)
#       visit_day_last      0.8679   (+0.0098)
#       late_days_mean      0.8722   (+0.0044)
#       clicks_homepage     0.8751   (+0.0029)
#       highest_education   0.8773   (+0.0022)
#       score_std           0.8783   (+0.0010)
#       clicks_dualpane     0.8796   (+0.0013)
#
# Each one was then re-tested on the validation slice, which the
# selection never saw (check_indicators.py,
# results_indicator_check.csv). score_std and clicks_dualpane each cost
# 0.0005 there and were removed, which is why this list holds six.
#
# No column about who the student is was chosen: gender, region,
# deprivation band, age band and disability were all offered and none
# was kept. highest_education is academic background, not a personal
# characteristic.
INDICATORS = [
    "score_mean",         # the mean of his marks so far
    "weight_sum",         # the weight of the work he has handed in
    "visit_day_last",     # the day of his last visit to the platform
    "late_days_mean",     # how late, on average, his work was
    "clicks_homepage",    # his clicks on the course home page
    "highest_education",  # the highest qualification he holds
]

# Kept in the processed files so the model can be audited, and never
# given to it. At 0.20 on the test slice the widest recall gap is 0.026
# by gender and 0.005 by age band, but 0.104 by region and 0.121 by
# deprivation band. The model reads none of these columns: the
# unevenness is in the study behaviour, which differs between regions.
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
