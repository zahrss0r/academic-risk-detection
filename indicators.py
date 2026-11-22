"""
indicators.py
Smart Academic Risk Detection System

Turns a student's record into the numbers the model reads.

WHAT IT BUILDS AND WHAT THE MODEL READS
    This file builds the whole candidate set that the selection experiment
    evaluated, and settings.INDICATORS names the six of them the model was
    trained on. The two are kept apart on purpose: the selection has to be
    able to measure a candidate in order to confirm or refuse it, and
    check_indicators.py reads the columns built here to do exactly that.
    for_the_model below hands the model only the names in
    settings.INDICATORS, so a column built here and not named there is
    computed and never shown to it.

WHY THIS FILE EXISTS
    The same arithmetic is needed in two places. process_data.py needs it to
    build the training data from the published tables, and app.py needs it to
    read the file an advisor uploads. If it were written twice, one copy
    would drift, and the model would be handed numbers built differently
    from the ones it was trained on. It is written once, here.

WHAT A RECORD LOOKS LIKE GOING IN
    One row per event, the way a university system exports its records:

      id_student  course      event_type   item_id  date_due  date_submitted  score  weight  activity_type  date  clicks
      653440      EEE-2014J   enrolment
      653440      EEE-2014J   submission   30719    33        32              88     16
      653440      EEE-2014J   submission   30720    68        68              93     28
      653440      EEE-2014J   activity     832565                                            oucontent      0     6
      653440      EEE-2014J   activity     832760                                            homepage       0     1

    A student has one enrolment row, a handful of submission rows and
    hundreds of activity rows. The model needs one row, so these are
    collapsed into one. The enrolment row is what says the student is
    registered at all, so a student with no other row is still read.

AND WHAT COMES OUT
    One row per student per course, with the candidate columns. Anything
    that happened after the cut day is thrown away first, and anything the
    file does not carry is left empty rather than filled with a number
    nobody measured: the model was trained to read an empty cell as "not
    known" and to decide for itself which way to send it.

    An empty cell is not a gap in this data. It is the single strongest
    thing the data says: of the students who had submitted nothing at all
    by the cut day, 98.9% went on to fail or withdraw.
"""

import pandas as pd
import numpy as np

import settings


KEYS = settings.KEY_COLUMNS
CUT_DAY = settings.CUT_DAY

# The kinds of page whose clicks are built. All sixteen kinds in the data
# were offered to the selection, and these two are the ones its forward path
# proposed; which of them the model actually reads is settled by
# check_indicators.py and named in settings.INDICATORS.
PAGE_KINDS = ["homepage", "dualpane"]


def from_events(events, students=None, cut_day=CUT_DAY):
    """
    Build the candidate columns from a table of events.

    events    one row per event, with the columns described above
    students  optional. One row per student, carrying the attributes that
              are not events - highest_education. If it is not given, the
              cohort is read from the enrolment rows of the event table.
    cut_day   nothing dated after this day is used

    Returns one row per student per course.
    """
    events = events.copy()

    # ---- who is in this cohort
    #
    # The list of who to judge comes from the enrolment rows, and only from
    # them. It is not gathered from whoever happens to appear in the
    # submission and activity rows, because a student who has handed in
    # nothing and never opened the platform appears in neither - and that
    # student is the one the system exists to catch. Taking the list from
    # the enrolment rows turns their silence into a reading of them instead
    # of their disappearance.
    if students is not None:
        out = students.copy()
    else:
        keep = [c for c in KEYS + ["highest_education"] if c in events.columns]
        enrolled = events[events["event_type"] == settings.EVENT_ENROLMENT]
        if len(enrolled):
            out = enrolled[keep].drop_duplicates(subset=KEYS)
        else:
            # A file with no enrolment rows is still read rather than
            # refused, from whoever its event rows mention. The screen says
            # that this is what happened, because a cohort read this way is
            # missing exactly the students who are in most trouble.
            out = events[keep].drop_duplicates(subset=KEYS)
    out = out.reset_index(drop=True)

    # ---- the work handed in
    work = events[events["event_type"] == settings.EVENT_SUBMISSION]
    if len(work) and "date_due" in work.columns:
        work = work[(pd.to_numeric(work["date_due"], errors="coerce") <= cut_day)
                    & (pd.to_numeric(work["date_submitted"],
                                     errors="coerce") <= cut_day)]
    if len(work):
        work = work.copy()
        work["score"] = pd.to_numeric(work["score"], errors="coerce")
        work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
        work["late_days"] = (pd.to_numeric(work["date_submitted"],
                                           errors="coerce")
                             - pd.to_numeric(work["date_due"], errors="coerce"))
        # A total over a group whose values are all empty is not zero -
        # it is not known, and the two are different facts here. A mean
        # already answers that way; a sum does not, and returns 0 unless it
        # is told to require at least one value. Without min_count a
        # student whose submission weights cannot be read is handed to the
        # model as one who submitted nothing of weight, which is the
        # strongest signal of failure in the data and would be invented
        # rather than measured.
        made = work.groupby(KEYS).agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            weight_sum=("weight", lambda column: column.sum(min_count=1)),
            late_days_mean=("late_days", "mean"),
        ).reset_index()
        out = out.merge(made, on=KEYS, how="left")
    else:
        for name in ["score_mean", "score_std", "weight_sum",
                     "late_days_mean"]:
            out[name] = np.nan

    # ---- what he did on the platform
    seen = events[events["event_type"] == settings.EVENT_ACTIVITY]
    if len(seen) and "date" in seen.columns:
        seen = seen.copy()
        seen["date"] = pd.to_numeric(seen["date"], errors="coerce")
        seen["clicks"] = pd.to_numeric(seen["clicks"], errors="coerce")
        seen = seen[(seen["date"] >= 0) & (seen["date"] <= cut_day)]
    if len(seen):
        last = seen.groupby(KEYS)["date"].max().rename(
            "visit_day_last").reset_index()
        out = out.merge(last, on=KEYS, how="left")
        for kind in PAGE_KINDS:
            part = seen[seen["activity_type"] == kind]
            name = "clicks_" + kind
            if len(part):
                made = (part.groupby(KEYS)["clicks"].sum()
                            .rename(name).reset_index())
                out = out.merge(made, on=KEYS, how="left")
            else:
                out[name] = np.nan
    else:
        out["visit_day_last"] = np.nan
        for kind in PAGE_KINDS:
            out["clicks_" + kind] = np.nan

    # ---- every indicator the model expects must be there, even if empty
    for name in settings.INDICATORS:
        if name not in out.columns:
            out[name] = np.nan

    return out


def which_are_empty(table):
    """
    Which indicators this cohort has no value for, and for how many rows.

    An advisor whose file is missing something should be told, not left to
    wonder why the system is quiet about half his students.
    """
    found = {}
    for name in settings.INDICATORS:
        if name not in table.columns:
            found[name] = len(table)
        else:
            empty = int(table[name].isna().sum())
            if empty:
                found[name] = empty
    return found


def for_the_model(table, education_levels=None):
    """
    The six columns in the order the model expects, typed the way it was
    trained on them.

    One of the six is text - the student's highest qualification - and
    LightGBM reads it as a category. The other five are numbers. Text in a
    numbered column is read the same way an empty cell is read: not known.
    That is the system's own answer for a value it does not have, and it is
    applied here rather than crashing the page.

    WHY education_levels IS PASSED IN
        A category column is not really text to the model; it is a position
        in a list. If the list is built fresh from whatever cohort happens
        to be uploaded, the same qualification can land in a different
        position than it had during training, and the model would then be
        reading one qualification as another - silently, with no error.

        So the list is built once, from the training data, saved beside the
        model, and handed back in here. A qualification the training data
        never contained becomes empty, which is the honest answer: the model
        was never taught what to do with it.

        It is optional only so that the training script itself can call this
        before the list exists.
    """
    frame = table.reindex(columns=settings.INDICATORS).copy()
    for name in frame.columns:
        if name == "highest_education":
            if education_levels is not None:
                frame[name] = pd.Categorical(frame[name],
                                             categories=education_levels)
            else:
                frame[name] = frame[name].astype("category")
        else:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame
