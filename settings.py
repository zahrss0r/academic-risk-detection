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

# A student may file an excuse or withdraw from the first day of term until
# two weeks before the final exam of that course.
#
# This used to be written here as a fixed day of term, day 90, which was
# wrong twice over. It is not our university's rule, and it cannot be a
# fixed day at all: every course sits its final on its own date, so the
# window closes on a different day in every course.
#
# So the rule needs the date of that course's final. Where a dataset
# carries it, the rule works out the closing day and says how long is left.
# Where it does not, the rule says nothing. It is not disabled and it is not
# guessed; it waits for the one fact it needs.
WINDOW_NOTICE_DAYS = 14     # the window shuts this long before the final

# Warn when the window is about to close within this many days.
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
# 300 each. At that size about 116 of the students shown will genuinely be
# at risk — 117, 114 and 118 in the three sets as drawn — which keeps the
# recall seen live on screen within roughly four points of the figure in the
# report. At 200 students it swings by five points or more, which invites an
# awkward question about why the screen and the report disagree.
DEMO_STUDENTS_PER_SET = 300
DEMO_SETS = ["supervisor", "final", "backup"]

# Raise an alert above this probability. Not the usual 0.50: missing a
# student costs that student a term, a false alarm costs a phone call, so
# the two mistakes are not worth the same. See results_threshold_table.csv.
ALERT_THRESHOLD = 0.30

# There is one threshold and no bands above it. An earlier version sorted
# students into high, medium and stable on a second threshold of 0.65. That
# was dropped: a band is a judgement the system is not entitled to make, and
# the advisor was being handed three words where the honest answer is one
# line - this student was flagged, and here is why.

# The eleven indicators the model is trained on, in the order it expects
# them. Two of them, the ones ending in _missing, are not measurements. Each
# says whether the indicator beside it was left empty, because two of the
# indicators are a division and a division has no answer when the bottom
# half is zero. Leaving the cell empty and recording that we left it empty
# keeps a fact that writing a number would have hidden: a student who handed
# in nothing has no lateness to measure, and he is not the same student as
# one who handed everything in on time.
MODEL_INDICATORS = [
    "early_score",
    "early_score_missing",
    "submission_rate",
    "late_rate",
    "late_rate_missing",
    "total_clicks",
    "activity_days_rate",
    "longest_gap_days",
    "num_of_prev_attempts",
    "studied_credits",
    "late_registration",
]

# The same indicators split into the groups the ablation experiment compares,
# plus the group we refuse. Training on the first group tells us what we gave
# up by refusing it, which turns a principle into a number we can report.
GROUP_DEMOGRAPHIC = [
    "gender",
    "region",
    "age_band",
    "imd_band",
    "highest_education",
    "disability",
]
GROUP_ACADEMIC = [
    "early_score",
    "early_score_missing",
    "submission_rate",
    "num_of_prev_attempts",
    "studied_credits",
    "late_registration",
]
GROUP_BEHAVIOURAL = [
    "late_rate",
    "late_rate_missing",
    "total_clicks",
    "activity_days_rate",
    "longest_gap_days",
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
# Neither has a rule of its own in university_rules.py yet. The cumulative
# average is collected by the interface and shown to the advisor, and nothing
# in the code reads it. Attendance is not collected at all: check_absence
# counts sessions missed, which is a different quantity from the percentage
# attended, and that rule is silent here because this data has no absence
# column. Saying either of them is "applied as a rule" would not be true.
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
