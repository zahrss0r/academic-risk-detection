# settings.py
#
# Every number and every list that the rest of the project depends on.
#
# There is one reason this file exists: if a number lives in five files,
# then one day somebody changes four of them. Here it is changed once and
# every file sees it.

import os


# ---------------------------------------------------------------
# Finding the project's own files
# ---------------------------------------------------------------

# The folder this file is sitting in. Every data file the system reads is
# beside it.
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def project_file(name):
    # The full path to a file that lives beside the code.
    #
    # We do not just write "final_model.pkl" and hope, because a program is
    # not always started from the folder its code is in. Streamlit Cloud
    # starts the application from the top of the repository, so a bare file
    # name would be looked for in the wrong place and nothing would load.
    # Building the path from this file's own location works wherever the
    # program is started from.
    return os.path.join(PROJECT_FOLDER, name)


# ---------------------------------------------------------------
# The marks, taken from the real assessment plan of the course
# ---------------------------------------------------------------

# Coursework, 40 marks in total.
COURSEWORK_PLAN = {
    "assignment_1": 8,
    "assignment_2": 8,
    "discussion": 4,
    "quiz": 6,
    "project": 14,
}

MIDTERM_MAX = 20
FINAL_MAX = 40
PASS_MARK = 60            # out of 100, below this is a fail


# ---------------------------------------------------------------
# The university rules
# ---------------------------------------------------------------

MAX_ABSENCE_IN_PERSON = 4   # unexcused, per course
MAX_ABSENCE_ONLINE = 4      # unexcused, per course

# The last day a student may still file an excuse or withdraw. It runs from
# the first day of term until just before the final exams, about three
# months.
WINDOW_LAST_DAY = 90

# Warn this many days before that window closes.
WINDOW_WARNING_DAYS = 14

# Warn when the highest total the student can still reach is below this.
# The pass mark is 60, so this is a margin of 10 marks.
TIGHT_MARGIN = 70


# ---------------------------------------------------------------
# The model
# ---------------------------------------------------------------

RANDOM_SEED = 42

# The prediction moment. Day 100 of the term, roughly the middle, just after
# the midterm. Nothing dated after this day is used anywhere.
CUT_DAY = 100

# The demonstration sets.
#
# Three separate groups of students, drawn at random and kept out of the
# training set, the validation set AND the test set. So a student shown on
# screen in any demonstration has never been seen by the model and is not
# part of any number we report.
#
# Three because there are three occasions: the supervisor meeting, the
# committee, and a spare in case something goes wrong on the day. They are
# the same size and drawn the same way, so any one of them can stand in for
# any other.
#
# 300 each. At that size roughly 137 of the students shown will genuinely
# be at risk, which keeps the recall seen live on screen within about three
# and a half points of the figure reported in the report. At 200 it swings
# by four points or more, which invites an awkward question about why the
# screen and the report disagree.
DEMO_STUDENTS_PER_SET = 300
DEMO_SETS = ["supervisor", "final", "backup"]

# Raise an alert above this probability. Not the usual 0.50: missing a
# student costs that student a term, a false alarm costs a phone call, so
# the two mistakes are not worth the same. See results_threshold_table.csv.
ALERT_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.65

# The nine indicators the model is trained on, in the order it expects them.
MODEL_INDICATORS = [
    "early_score",
    "submission_rate",
    "late_rate",
    "total_clicks",
    "activity_days_rate",
    "longest_gap_days",
    "num_of_prev_attempts",
    "studied_credits",
    "late_registration",
]

# Kept in the processed files so we can check the model afterwards, and
# never given to it. A model that cannot see gender cannot learn to treat
# students differently because of it, and keeping the column lets us check
# that it did not reach the same bias by another route.
AUDIT_ONLY_COLUMNS = [
    "gender",
    "region",
    "age_band",
    "imd_band",
    "highest_education",
    "disability",
    "final_result",
]

# Two indicators our university relies on that no dataset we could obtain
# carries alongside the rest. They are NOT rejected. The data the model
# learned from has no column for either, so the model was never taught to
# read them, and handing a trained model a number it has never seen gives
# noise instead of a prediction.
#
# Both are collected by the interface and both are applied as rules in
# university_rules.py, which is how the university applies them anyway.
#
# WHEN DATA ARRIVES THAT CONTAINS THEM:
#   1. move the two names from here into MODEL_INDICATORS above
#   2. make sure the training file has a column with each of those names
#   3. in app.py, add them to the dictionary in to_model_indicators
#   4. run train_models.py again, then evaluate.py
WAITING_FOR_DATA = [
    "cumulative_gpa",     # 0.00 to 4.00
    "attendance_rate",    # percentage of sessions attended in this course
]


# ---------------------------------------------------------------
# Converting our university's numbers into the model's units
# ---------------------------------------------------------------

# A normal full time load. The Open University counts a normal year as 60
# credits, we count a normal term as about 15, because a course is 3 credits
# and a student usually takes five. So one of our credits stands for four of
# theirs.
#
# This is the one conversion in the whole project that is a judgement rather
# than a fact. We are saying a full load here means the same pressure as a
# full load there. Reasonable, not proven, and the first thing to check
# against real data from our own university.
OUR_FULL_LOAD = 15
THEIR_FULL_LOAD = 60
CREDIT_FACTOR = THEIR_FULL_LOAD / OUR_FULL_LOAD
