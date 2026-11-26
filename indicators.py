"""
indicators.py
Smart Academic Risk Detection System

Turns a student's event records into the columns the model reads.

Input is one row per event: one enrolment row per student per course, a
few submission rows, and many activity rows. Output is one row per
student per course.

This file builds the whole candidate set the selection experiment used.
settings.INDICATORS names the six the model was trained on, and
for_the_model hands it only those.

Both process_data.py and app.py call it, so the training data and an
uploaded cohort are built by the same arithmetic.

Events after the cut day are dropped. A column the file gives nothing
for is left empty, not set to zero.
"""

import pandas as pd
import numpy as np

import settings


KEYS = settings.KEY_COLUMNS
CUT_DAY = settings.CUT_DAY

# The kinds of page whose clicks are built. Every kind the data carries
# was offered to the selection; these two are the ones it proposed.
PAGE_KINDS = ["homepage", "dualpane"]


def from_events(events, students=None, cut_day=CUT_DAY):
    """Build the candidate columns from a table of events.

    events    one row per event
    students  optional, one row per student with highest_education. If not
              given, the cohort is read from the enrolment rows.
    cut_day   events after this day are dropped

    Returns one row per student per course.
    """
    events = events.copy()

    # A column the file does not carry is added empty, so the indicator built
    # from it comes out empty rather than stopping the scan.
    for name in settings.EVENT_COLUMNS:
        if name != "event_type" and name not in events.columns:
            events[name] = np.nan

    # ---- who is in this cohort
    #
    # The list comes from the enrolment rows only. A student who handed in
    # nothing and never opened the platform appears in no other row, and
    # that student is the one the system exists to catch.
    if students is not None:
        out = students.copy()
    else:
        keep = [c for c in KEYS + ["highest_education"] if c in events.columns]
        enrolled = events[events["event_type"] == settings.EVENT_ENROLMENT]
        if len(enrolled):
            out = enrolled[keep].drop_duplicates(subset=KEYS)
        else:
            # A file with no enrolment rows is still read, from whoever its event
            # rows mention. The screen says so.
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
        # A sum over a group whose values are all empty returns 0 unless it is
        # told to require at least one value. Without min_count a student whose
        # weights cannot be read would be handed to the model as one who
        # submitted nothing of weight.
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

    # ---- activity on the platform
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
                made = (part.groupby(KEYS)["clicks"].sum(min_count=1)
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


def missing_keys(events):
    """The columns a cohort file must carry before anything can be read.

    Without a student number there is nobody to register or notify.
    Without event_type no row can be told from another. These are
    reported, never filled in.
    """
    found = [name for name in settings.KEY_COLUMNS
             if name not in events.columns]
    if "event_type" not in events.columns:
        found = found + ["event_type"]
    return found


def which_are_empty(table):
    """Which indicators this cohort has no value for, and for how many rows."""
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
    """The six columns in the order the model expects, typed as trained.

    One column is text - the highest qualification - and LightGBM reads it
    as a category. Text in a numbered column is read as not known.

    education_levels is the qualification list saved with the model. A
    category is a position in a list, so a list rebuilt from the uploaded
    cohort could put the same qualification in a different position and
    the model would read one as another. A qualification the training
    data never held becomes empty. It is optional only so the training
    script can call this before the list exists.
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
