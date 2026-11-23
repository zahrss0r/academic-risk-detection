"""
settings.py
Smart Academic Risk Detection System

Every number and every list that the rest of the system depends on.

There is one reason this file exists: if a number lives in five files, then
one day somebody changes four of them. Here it is changed once and every
file sees it.

The system is one trained model. It reads a cohort's records, builds six
indicators from them, gives each student a probability, and raises an alert
above a threshold. Each of those three things - the six indicators, the day
the scan happens, and the threshold - was measured, and the measurement is
written beside it below.
"""

import os


# ---------------------------------------------------------------
# Finding the project's own files
# ---------------------------------------------------------------

# The folder this file is sitting in. Every data file the system reads is
# beside it.
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def project_file(name):
    """
    The full path to a file that lives beside the code.

    We do not write a bare file name and hope, because a program is not
    always started from the folder its code is in. Streamlit Cloud starts
    the application from the top of the repository, so a bare name would be
    looked for in the wrong place and nothing would load. Building the path
    from this file's own location works wherever the program is started
    from.
    """
    return os.path.join(PROJECT_FOLDER, name)


# ---------------------------------------------------------------
# The model
# ---------------------------------------------------------------

# Fixed so that every run of every script in this project gives the same
# answer, and anybody re-running them gets our numbers and not numbers
# close to ours.
RANDOM_SEED = 42

# The prediction moment: day 110 of the course run.
#
# The day was read off the courses themselves, under two conditions that
# have to hold at the same time:
#
#   at least two weighted pieces of work have already fallen due, so there
#   is something to judge the student on, and
#   at least two are still ahead, so there is still something to predict.
#
# Those two conditions leave a window of day 68 to day 116 across the
# fourteen course runs the model is trained on. Day 110 sits near the top of
# that window, where 43 per cent of the coursework weight is visible rather
# than the 25 per cent visible at day 68. A course run in this data lasts
# between 234 and 269 days, and the assessments in it are tutor-marked work,
# computer-marked work and a final exam.
#
# Nothing dated after this day is used anywhere in the system.
CUT_DAY = 110

# Raise an alert above this probability.
#
# Not the usual 0.50. Missing a student costs that student a term; a false
# alarm costs the advisor a phone call, and what to do about an alert is the
# advisor's decision and not the system's. The two mistakes are not worth
# the same, so the threshold is not left at its default.
#
# At 0.20, on the held-out test slice: recall 0.8984, precision 0.6159, 91
# students missed out of 896, and 502 false alarms. These are the figures in
# results_final.csv, written by the single opening of the test slice. The
# whole table of eleven thresholds is in results_threshold_table.csv, on the
# validation slice where the threshold was chosen, and the report prints all
# of it rather than this row alone.
#
# A low threshold also turned out to be the fairer one. On the validation
# slice, where every threshold was tried, the widest gap in recall between
# regions is 0.161 at 0.20 against 0.189 at 0.30, and the gap on disability
# widens steadily as the threshold is tightened. On the test slice at 0.20
# the same gap is 0.104 by region and 0.121 by deprivation band. Both slices
# are named because the figures differ between them, and only the validation
# figures can be checked against results_fairness.csv, which is the file
# that ships.
ALERT_THRESHOLD = 0.20

# There is one threshold and no bands above it. Sorting students into high,
# medium and stable would be a judgement the system is not entitled to make,
# and it would hand the advisor three words where the honest answer is one
# line: this student was flagged, and here is why.

# The six indicators the model is trained on, in the order it expects them.
#
# These were not chosen by us. Every column in the raw data was offered to a
# forward-selection experiment - seventy-one candidates, counting the
# original columns and every standard aggregation of the dated tables - and
# the experiment kept the one that helped most at each step until helping
# stopped. The stopping rule was declared before the run: stop at the last
# step that raised cross-validated ROC-AUC by 0.001 or more.
#
# The curve it produced, from nothing to the end of the path:
#       score_mean          0.8130   (+0.3130)
#       weight_sum          0.8581   (+0.0451)
#       visit_day_last      0.8679   (+0.0098)
#       late_days_mean      0.8722   (+0.0044)
#       clicks_homepage     0.8751   (+0.0029)
#       highest_education   0.8773   (+0.0022)
#       score_std           0.8783   (+0.0010)
#       clicks_dualpane     0.8796   (+0.0013)
# and the ninth step lowered the figure by 0.0002, which is where the rule
# stopped the path - at eight steps, not at the six listed below.
#
# Every candidate the path kept was then checked a second time on the
# validation slice, which the selection never saw: train with it, train
# without it, and compare both the ranking and the students whose decision
# moves. A candidate earns its place only if it raises ROC-AUC on data it
# was not chosen on. The check is check_indicators.py and its table is
# results_indicator_check.csv.
#
# Two of the eight did not earn it. score_std and clicks_dualpane each cost
# 0.0005 on the validation slice when added, so the forward path kept them
# and the second check removed them. That is why this list holds six while
# the path held eight: the rule above says where the path stopped, and the
# removal check says what survived it.
#
# Two things worth knowing about the list. One indicator carries most of it:
# score_mean alone takes the model from 0.50 to 0.813, and the other five
# add the rest. And six beat the eight the path had kept: 0.8714 against
# 0.8699 on the validation slice, because the last two were the same fact
# told a second way.
#
# WHAT THE EXPERIMENT DID NOT CHOOSE
#   Not one column about who the student is. Gender, region, deprivation
#   band, age band and disability were all offered, and the path took none
#   of them. Neither did it take the course code or the length of the run.
#   The one original column that survived is highest_education, which is
#   academic background rather than a personal characteristic.
INDICATORS = [
    "score_mean",         # the mean of his marks so far
    "weight_sum",         # the weight of the work he has handed in
    "visit_day_last",     # the day of his last visit to the platform
    "late_days_mean",     # how late, on average, his work was
    "clicks_homepage",    # his clicks on the course home page
    "highest_education",  # the highest qualification he holds
]

# Kept in the processed files so the model can be checked afterwards, and
# never given to it. A model that cannot see gender cannot learn to treat
# students differently because of it, and keeping the column lets us check
# whether it reached the same unevenness by another route.
#
# It does, partly, and the figures say so: at a threshold of 0.20 on the
# test slice the widest gap in recall is 0.026 by gender and 0.005 by age
# band, but 0.104 by region and 0.121 by deprivation band. On the validation
# slice, which is the one results_fairness.csv carries, the same gaps are
# 0.029 by gender, 0.020 by age band, 0.161 by region and 0.096 by
# deprivation band. The model was given none of those columns. The
# unevenness is in the study behaviour itself, which differs between
# regions, and the report states it as it is.
AUDIT_ONLY_COLUMNS = [
    "gender",
    "region",
    "age_band",
    "imd_band",
    "disability",
    "final_result",
]

# Which event rows an uploaded cohort file can carry, and what each kind is
# read for. The file is one row per event, the way a university system
# exports its records, and the system does the rest: it cuts everything
# after the cut day, builds the six indicators, scores, and reports.
EVENT_ENROLMENT = "enrolment"     # one row per student per course
EVENT_SUBMISSION = "submission"   # one piece of work handed in
EVENT_ACTIVITY = "activity"       # one day's clicks on one page

# WHY AN ENROLMENT ROW EXISTS AND IS NOT OPTIONAL
#   A student who has handed in nothing and never opened the platform has
#   no submission row and no activity row, so a file built only of those
#   two kinds has no row for that student anywhere - and the system would
#   never know they were registered. They are not a rare case and they are
#   not a safe one to lose: in the training data 191 of 9,761 records are
#   students with no event of any kind by the cut day, and 99.0 per cent of
#   them did not finish.
#
#   The enrolment row is what makes the cohort a cohort. It says this
#   student is registered on this course, and it carries the few things
#   about a student that are not events. The system builds its list of who
#   to judge from those rows, so silence about a student becomes a reading
#   of that student rather than their disappearance.

# The columns an event file is read for. A file that does not carry one of
# them is not rejected: the indicator built from it is left empty, and the
# model was trained to read an empty cell as "not known" and to decide which
# way to send it.
#
# event_type is the exception, and the scan stops without it. It is not a
# column an indicator is built from - it is the column that says what each
# row is, so without it there is no submission to read and no visit to
# count, and every indicator would be empty at once. A file that does not
# carry it is named on the screen and the scan does not run.
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
