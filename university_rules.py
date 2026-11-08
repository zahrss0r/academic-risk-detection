# university_rules.py
#
# The first half of the system: the university's written rules, applied with
# ordinary arithmetic. There is no machine learning in this file on purpose.
#
# Four unexcused absences means the student is barred from the course. That
# is a fact to be counted, not a probability to be estimated, and a system
# that guesses at a fact it could simply count is a worse system.
#
# The second half is the trained model, for the students no rule has caught.

import settings


def make_alert(rule, level, message):
    # Every rule returns alerts in this shape, so the interface can display
    # any of them without knowing which rule produced it.
    # level is "barred", "warning" or "information".
    return {"rule": rule, "level": level, "message": message}


# ---------------------------------------------------------------
# The marks
# ---------------------------------------------------------------

def coursework_earned(student):
    # Adds up the coursework the student has a mark for. Work that was due
    # and not handed in has no mark, so it adds nothing, which is correct:
    # it scored zero.
    total = 0
    for part in settings.COURSEWORK_PLAN:
        mark = student["marks"].get(part)
        if mark is not None:
            total = total + mark
    return total


def coursework_still_available(student):
    # Adds up only the coursework that has not fallen due yet.
    #
    # This is the careful part. Work that was due and was not handed in is a
    # zero the student already owns, not a mark they can still go and earn,
    # so it must not be counted here. Counting it would let the system tell
    # an advisor that a student who submitted nothing can still score 100.
    total = 0
    for part in settings.COURSEWORK_PLAN:
        has_mark = student["marks"].get(part) is not None
        has_fallen_due = part in student["due_so_far"]
        if not has_mark and not has_fallen_due:
            total = total + settings.COURSEWORK_PLAN[part]
    return total


def midterm_earned(student):
    # The mark if it was sat, zero if it was held and missed.
    if student["midterm"] is not None:
        return student["midterm"]
    return 0


def midterm_still_available(student):
    # Only still winnable if the midterm has not been held yet.
    if student["midterm"] is None and not student["midterm_was_held"]:
        return settings.MIDTERM_MAX
    return 0


def mark_so_far(student):
    # What the student has actually earned today, out of 100.
    return coursework_earned(student) + midterm_earned(student)


def highest_possible_mark(student):
    # The best final total the student can still reach, scoring full marks
    # on everything that is left. This is the most useful single number an
    # advisor can be given: once it drops below 60 the course is lost no
    # matter what the student does next.
    still_available = (coursework_still_available(student)
                       + midterm_still_available(student)
                       + settings.FINAL_MAX)
    return mark_so_far(student) + still_available


# ---------------------------------------------------------------
# The rules
# ---------------------------------------------------------------

def check_absence(student):
    # Rule one. Four unexcused absences in person, or four online, bars the
    # student. We warn one absence before the limit, because the point of
    # the system is to reach the student while something can still be done.
    alerts = []

    pairs = [
        ("absence in person", student["absence_in_person"], settings.MAX_ABSENCE_IN_PERSON),
        ("absence online", student["absence_online"], settings.MAX_ABSENCE_ONLINE),
    ]

    for name, missed, limit in pairs:
        if missed >= limit:
            alerts.append(make_alert(name, "barred",
                                     "Barred from the course. " + str(missed)
                                     + " unexcused absences, the limit is "
                                     + str(limit) + "."))
        elif missed == limit - 1:
            alerts.append(make_alert(name, "warning",
                                     "One absence away from being barred. "
                                     + str(missed) + " of " + str(limit) + "."))
    return alerts


def check_marks(student):
    # Rule two. The pass mark is 60. We compare the highest total the
    # student can still reach against it.
    highest = highest_possible_mark(student)

    if highest < settings.PASS_MARK:
        return [make_alert("cannot reach the pass mark", "barred",
                           "Cannot pass. The highest total still reachable is "
                           + str(round(highest, 1)) + " and the pass mark is "
                           + str(settings.PASS_MARK) + ".")]

    if highest < settings.TIGHT_MARGIN:
        return [make_alert("very little margin left", "warning",
                           "Only " + str(round(highest - settings.PASS_MARK, 1))
                           + " marks of margin above the pass mark. Near perfect "
                           + "scores are needed on everything that is left.")]
    return []


def check_submissions(student):
    # Rule three. Work that was due and not submitted scores zero. The
    # student may still file an excuse, so we name the components instead of
    # only reporting a low total.
    alerts = []

    missing = []
    for part in student["due_so_far"]:
        if student["marks"].get(part) is None:
            missing.append(part)

    if len(missing) > 0:
        lost = 0
        for part in missing:
            lost = lost + settings.COURSEWORK_PLAN[part]
        alerts.append(make_alert("not submitted", "warning",
                                 "Nothing submitted for " + ", ".join(missing)
                                 + ". That is " + str(lost)
                                 + " marks scored as zero unless an excuse is filed."))

    if student["midterm"] is None and student["midterm_was_held"]:
        alerts.append(make_alert("missed the midterm", "warning",
                                 "The midterm was held and no mark was recorded. That is "
                                 + str(settings.MIDTERM_MAX)
                                 + " marks scored as zero unless an excuse is filed."))
    return alerts


def check_window(student):
    # Rule four. How long the student still has to file an excuse or
    # withdraw.
    #
    # The system never advises a student to withdraw. That decision belongs
    # to the student and the advisor. All we do is say how much time is
    # left, because an alert that arrives after the window has closed is
    # useless.
    #
    # The window shuts two weeks before that course's final exam, so this
    # rule needs the date of the final. Every course sits its own, and no
    # fixed day of term can stand in for it.
    exam_day = student.get("exam_day")

    if exam_day is None:
        # The one fact this rule needs is not in front of it, so it says
        # that and nothing else. A silent rule is not a broken rule. The
        # alternative would be to assume a date, which would put a number
        # on the screen that nothing in the data supports.
        return [make_alert("window unknown", "information",
                           "The date of the final exam is not in this data, "
                           "so the withdrawal window cannot be worked out.")]

    closes_on = exam_day - settings.WINDOW_NOTICE_DAYS
    days_left = closes_on - student["day_of_term"]

    if days_left <= 0:
        return [make_alert("window closed", "information",
                           "The window for filing an excuse or withdrawing "
                           "closed on day " + str(closes_on) + ".")]

    if days_left <= settings.WINDOW_WARNING_DAYS:
        return [make_alert("window closing", "warning",
                           str(days_left) + " days left to file an excuse or "
                           "withdraw. The window shuts on day "
                           + str(closes_on) + ".")]

    return [make_alert("window open", "information",
                       str(days_left) + " days left to file an excuse or "
                       "withdraw. The window shuts on day "
                       + str(closes_on) + ".")]


# ---------------------------------------------------------------
# Running all of them
# ---------------------------------------------------------------

def policy_summary(student):
    # Runs every rule and returns one answer for the interface.
    alerts = (check_absence(student)
              + check_marks(student)
              + check_submissions(student)
              + check_window(student))

    status = "clear"
    for alert in alerts:
        if alert["level"] == "barred":
            status = "barred"
        elif alert["level"] == "warning" and status == "clear":
            status = "warning"

    return {
        "status": status,
        "mark_so_far": mark_so_far(student),
        "highest_possible": highest_possible_mark(student),
        "alerts": alerts,
    }


# ---------------------------------------------------------------
# The same regulations, applied to a cohort file
# ---------------------------------------------------------------
#
# policy_summary above needs one student's full mark sheet: every component,
# whether the midterm was held, how many sessions were missed. That is what
# an advisor has in front of them, and it is what the university's own
# system would hand over.
#
# A cohort file is not that. It is one row per student per course, holding
# the indicators the system was built on, and it carries no per component
# marks and no absence column at all. So the rules cannot all be asked the
# same way, and pretending otherwise would mean inventing values.
#
# This function asks each rule what it can reach in such a row, and lets
# each one answer for itself. A rule with no column to read says so. None
# of them guesses.

def cohort_row_summary(row):
    """Apply the regulations to one row of a cohort file.

    row is anything that can be asked `row.get(name)`, such as a pandas
    row. Returns the same shape policy_summary returns, so the interface
    can show either without knowing which one it got.
    """
    alerts = []

    # Rule: work that was due and not handed in.
    #
    # The regulation says such work scores zero unless an excuse is filed.
    # A student who has not handed in everything that was due is exactly a
    # student whose submission_rate is below 100, so this rule can be
    # applied to a cohort file in full.
    handed_in = row.get("submission_rate")
    if handed_in is None or handed_in != handed_in:        # None, or NaN
        alerts.append(make_alert(
            "submissions unknown", "information",
            "This file does not say how much of the due work was handed "
            "in, so the submissions rule cannot be applied."))
    elif handed_in < 100:
        missed = round(100 - handed_in, 1)
        alerts.append(make_alert(
            "work not handed in", "warning",
            str(round(handed_in, 1)) + " percent of the work that was due "
            "was handed in, and " + str(missed) + " percent was not. What "
            "was not handed in scores zero unless an excuse is filed."))

    # Rule: unexcused absence.
    #
    # Silent here, and silent for a reason that is worth stating plainly:
    # the published data this system was built on has no absence column at
    # all. The rule is written and tested in test_system.py; it is waiting
    # for a column, not broken.
    if row.get("absence_in_person") is None:
        alerts.append(make_alert(
            "absence unknown", "information",
            "This file carries no record of absence, so the absence rule "
            "cannot be applied."))

    # Rule: the highest mark still reachable.
    #
    # This one needs the weight of every assessment in the course and the
    # marks earned so far, component by component. A cohort file carries a
    # single averaged early mark instead, so the ceiling cannot be worked
    # out from it and the rule stays silent rather than guess at one.
    if row.get("due_so_far") is None:
        alerts.append(make_alert(
            "ceiling unknown", "information",
            "This file does not carry the weight of each assessment, so "
            "the highest mark still reachable cannot be worked out."))

    # Rule: the excuse and withdrawal window.
    #
    # Needs the date of that course's final exam, which a cohort file does
    # not carry either.
    if row.get("exam_day") is None:
        alerts.append(make_alert(
            "window unknown", "information",
            "The date of the final exam is not in this file, so the "
            "withdrawal window cannot be worked out."))

    status = "clear"
    for alert in alerts:
        if alert["level"] == "barred":
            status = "barred"
        elif alert["level"] == "warning" and status == "clear":
            status = "warning"

    return {"status": status, "alerts": alerts}


def what_the_rules_can_reach(row):
    """One line per regulation, saying what it could do with this file.

    The interface shows this as a table so the committee can see at a
    glance which regulations spoke and which were waiting for a column.
    It is built from the alerts the rules themselves produced, not from a
    sentence written by hand, so it cannot drift away from what they did.
    """
    summary = cohort_row_summary(row)
    said = {}
    for alert in summary["alerts"]:
        said[alert["rule"]] = alert

    return [
        {"rule": "work not handed in", "needs": "submission_rate",
         "silent": "submissions unknown" in said},
        {"rule": "absence", "needs": "a count of unexcused absences",
         "silent": "absence unknown" in said},
        {"rule": "marks still reachable", "needs": "the weight of each assessment",
         "silent": "ceiling unknown" in said},
        {"rule": "the withdrawal window", "needs": "the date of the final exam",
         "silent": "window unknown" in said},
    ]
