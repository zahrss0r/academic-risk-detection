# app.py
#
# The screen the academic advisor uses.
#
# The system has two halves and this screen keeps them apart on purpose:
#
#   the policy engine   what the university rules already say about this
#                       student today, counted with plain arithmetic
#
#   the trained model   of the students no rule has caught yet, which ones
#                       look like the students who did not pass
#
# They are shown side by side rather than combined into one number, because
# they are different kinds of claim. One is counted from regulations and is
# certain. The other is estimated and is not. Averaging them together would
# let a certainty and a guess cancel each other out, and the advisor would
# not be able to tell which was which.

import streamlit as st
import pandas as pd
import joblib
from datetime import date

import settings
import university_rules
import database

FEATURES = settings.MODEL_INDICATORS

st.set_page_config(page_title="Academic Risk Detection", layout="wide")

model = joblib.load(settings.project_file("final_model.pkl"))

# The three demonstration sets. Every student in them was kept out of the
# training set, the validation set and the test set, so nothing shown on
# these screens has been seen by the model or counted in any reported
# number. See settings.py for why there are three.
demo_sets = {}
for name in settings.DEMO_SETS:
    demo_sets[name] = pd.read_csv(settings.project_file("demo_" + name + ".csv"))

DEMO_LABELS = {
    "supervisor": "Supervisor meeting",
    "final": "Final committee",
    "backup": "Spare set",
}

# Creates predictions.db the first time the application runs, and does
# nothing on every run after that.
database.create_tables()


# ---------------------------------------------------------------
# Turning the advisor's mark sheet into the model's nine indicators
# ---------------------------------------------------------------
#
# The model was trained on another university's data and speaks in that
# university's units. An advisor here has a mark sheet, an absence register
# and a platform report. These functions are the bridge, and every
# conversion is written out rather than hidden, because this is the part of
# the system that carries an assumption.

def safe_percent(top, bottom):
    # Divide and turn into a percentage, returning zero when the bottom is
    # zero instead of crashing.
    if bottom > 0:
        return round(top / bottom * 100, 2)
    return 0.0


def marks_assessed_so_far(student):
    # How many marks have actually been assessed today. This is the
    # denominator for the score, and it matters: 20 out of 25 assessed
    # marks is a student doing well, 20 out of 60 is not. Dividing both by
    # 100 would hide the difference completely.
    total = 0
    for part in student["due_so_far"]:
        total = total + settings.COURSEWORK_PLAN[part]
    if student["midterm_was_held"]:
        total = total + settings.MIDTERM_MAX
    return total


def to_model_indicators(student):
    # Builds the nine numbers the model expects.
    handed_in = 0
    for part in student["due_so_far"]:
        if student["marks"].get(part) is not None:
            handed_in = handed_in + 1

    return {
        # the percentage scored on the work assessed so far
        "early_score": safe_percent(university_rules.mark_so_far(student),
                                    marks_assessed_so_far(student)),

        # how much of the work that was due was handed in
        "submission_rate": safe_percent(handed_in, len(student["due_so_far"])),

        # how much of the handed in work arrived after the deadline
        "late_rate": safe_percent(student["late_submissions"], handed_in),

        # how many of the term's days so far the course was opened
        "activity_days_rate": safe_percent(student["active_days"],
                                           student["day_of_term"]),

        # the credit load, converted to the scale the model learned on
        "studied_credits": student["credits_this_term"] * settings.CREDIT_FACTOR,

        # these three mean the same thing in both places, so no conversion
        "total_clicks": student["total_clicks"],
        "longest_gap_days": student["longest_gap_days"],
        "num_of_prev_attempts": student["previous_attempts"],

        "late_registration": student["days_registered_late"],
    }


# The cumulative average and the attendance rate are collected on this
# screen but are not in the dictionary above, for one reason: the dataset
# the model learned from has no column for either, so the model was never
# taught to read them. Both are used by university_rules.py, where they are
# counted as rules instead of estimated. settings.WAITING_FOR_DATA lists
# the four steps that switch them on once real data carries them.


# ---------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------

def predict(values):
    one_row = pd.DataFrame([values])
    return model.predict_proba(one_row[FEATURES])[0][1]


def risk_level(probability):
    if probability >= settings.HIGH_RISK_THRESHOLD:
        return "HIGH RISK"
    if probability >= settings.ALERT_THRESHOLD:
        return "MEDIUM RISK"
    return "STABLE"


def combined_verdict(policy_status, level):
    # When the two halves disagree, the rule wins. A rule is something the
    # university has already decided; the model only offers a probability.
    if policy_status == "barred":
        return "error", "Act now. A university rule has already been triggered."
    if level == "HIGH RISK":
        return "error", ("Contact the student now. No rule is broken yet, "
                         "which is why the model matters here.")
    if policy_status == "warning" or level == "MEDIUM RISK":
        return "warning", "Keep this student under review."
    return "success", "No action needed."


def explain(values):
    # Plain language reasons the advisor can repeat to a student without
    # reading out a probability.
    reasons = []

    if values["early_score"] < 50:
        reasons.append("Scored " + str(values["early_score"])
                       + " percent of the marks assessed so far")
    if values["submission_rate"] < 100:
        reasons.append("Handed in only " + str(values["submission_rate"])
                       + " percent of the work that was due")
    if values["longest_gap_days"] >= 14:
        reasons.append("Did not open the course for "
                       + str(values["longest_gap_days"]) + " days in a row")
    if values["activity_days_rate"] < 20:
        reasons.append("Opened the course on only "
                       + str(values["activity_days_rate"])
                       + " percent of the days so far")
    if values["total_clicks"] < 200:
        reasons.append("Very little activity on the platform, only "
                       + str(values["total_clicks"]) + " clicks")
    if values["late_rate"] > 50:
        reasons.append("Handed in " + str(values["late_rate"])
                       + " percent of the work after the deadline")
    if values["num_of_prev_attempts"] > 0:
        reasons.append("Has taken this course before")

    if len(reasons) == 0:
        reasons.append("No warning sign in any indicator")
    return reasons


def show_table(file_name, caption):
    # Every result table on the performance tab is shown the same way.
    try:
        st.dataframe(pd.read_csv(settings.project_file(file_name)),
                     width="stretch")
        if caption != "":
            st.caption(caption)
    except Exception:
        st.info(file_name + " not found")


# ---------------------------------------------------------------
# The page
# ---------------------------------------------------------------

st.title("Smart Academic Risk Detection System")
st.caption("Prototype. The prediction is made for one student in one course "
           "at the middle of the term. It supports the academic advisor, it "
           "does not replace the advisor's decision.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Check a course", "A student's courses", "Model performance", "About"])


# ----- tab 1 : one student in one course -----
with tab1:
    # The verdict goes at the top, because that is what the advisor opens
    # the page to see. Streamlit runs a script from top to bottom, so we
    # reserve an empty container here and fill it further down, once the
    # numbers have been read and the model has been asked.
    verdict_box = st.container()

    st.subheader("What the advisor has in front of them")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Who and which course**")
        student_number = st.text_input("University number", "440012345")
        course_code = st.text_input("Course code", "DS483")
        program = st.text_input("Program", "Data Science")
        academic_level = st.number_input("Level", 1, 8, 6)

        st.markdown("**The term**")
        day_of_term = st.slider("Day of the term today", 1, 110, 60)
        credits_this_term = st.number_input(
            "Credits the student is taking this term", 3, 21, 15, step=3)
        previous_attempts = st.number_input(
            "Times this course was taken before", 0, 6, 0)
        days_registered_late = st.number_input("Days late in registering", 0, 60, 0)

        st.markdown("**Held for the rules, not sent to the model**")
        cumulative_gpa = st.number_input("Cumulative average (out of 4.00)",
                                         0.0, 4.0, 3.0, step=0.01)
        st.caption("Collected and shown to the advisor. Not a model input "
                   "yet, because no dataset we could obtain carries it "
                   "alongside the other indicators.")

    with col2:
        st.markdown("**Attendance register, unexcused only**")
        absence_in_person = st.number_input("In person sessions missed", 0, 20, 0)
        absence_online = st.number_input("Online sessions missed", 0, 20, 0)
        st.caption("An absence with an accepted excuse is not counted here. "
                   "The rule is about unexcused absence, not about an "
                   "attendance percentage.")

        st.markdown("**Learning platform**")
        total_clicks = st.number_input("Clicks on the course page", 0, 15000, 400)
        active_days = st.number_input("Days the course was opened", 0, 110, 20)
        longest_gap_days = st.slider(
            "Longest run of days without opening it", 0, 100, 10)

    with col3:
        st.markdown("**Mark sheet**")
        st.caption("Untick 'due' for work that has not fallen due yet. Untick "
                   "'handed in' for work that was due and never arrived, "
                   "which scores zero unless an excuse is accepted.")

        # The five coursework components as one small editable table. The
        # names and maximum marks come from settings.py, so the screen can
        # never drift away from the rules.
        plan_rows = []
        for part in settings.COURSEWORK_PLAN:
            maximum = settings.COURSEWORK_PLAN[part]
            plan_rows.append({
                "component": part.replace("_", " "),
                "out of": maximum,
                "due": True,
                "handed in": True,
                "mark": float(maximum) * 0.75,
            })

        sheet = st.data_editor(pd.DataFrame(plan_rows), hide_index=True,
                               width="stretch", disabled=["component", "out of"])

        # Read the table back into the two things the rules need: what has
        # fallen due, and what was scored on the parts handed in.
        marks = {}
        due_so_far = []
        part_names = list(settings.COURSEWORK_PLAN.keys())
        for position in range(len(sheet)):
            part = part_names[position]
            row = sheet.iloc[position]
            if row["due"]:
                due_so_far.append(part)
                if row["handed in"]:
                    marks[part] = row["mark"]

        midterm_was_held = st.checkbox("The midterm has been held", value=True)
        midterm = None
        if midterm_was_held:
            sat_the_midterm = st.checkbox("The student sat the midterm", value=True)
            if sat_the_midterm:
                midterm = st.number_input(
                    "Midterm mark (out of " + str(settings.MIDTERM_MAX) + ")",
                    0.0, float(settings.MIDTERM_MAX), 14.0, step=0.5)

        late_submissions = st.number_input(
            "How many of those were handed in late", 0, 10, 0)

    student = {
        "day_of_term": day_of_term,
        "absence_in_person": absence_in_person,
        "absence_online": absence_online,
        "midterm": midterm,
        "midterm_was_held": midterm_was_held,
        "marks": marks,
        "due_so_far": due_so_far,
        "late_submissions": late_submissions,
        "active_days": active_days,
        "total_clicks": total_clicks,
        "longest_gap_days": longest_gap_days,
        "previous_attempts": previous_attempts,
        "credits_this_term": credits_this_term,
        "days_registered_late": days_registered_late,
        "cumulative_gpa": cumulative_gpa,
    }

    summary = university_rules.policy_summary(student)
    values = to_model_indicators(student)
    probability = predict(values)
    level = risk_level(probability)
    kind, message = combined_verdict(summary["status"], level)

    # Filling the container reserved at the top of the tab.
    verdict_box.markdown("### What the system says")
    if kind == "error":
        verdict_box.error("**" + message + "**")
    elif kind == "warning":
        verdict_box.warning("**" + message + "**")
    else:
        verdict_box.success("**" + message + "**")

    left, right = verdict_box.columns(2)

    with left:
        st.subheader("1. University policy")
        st.caption("Counted directly from the rules. No model involved.")

        a, b = st.columns(2)
        a.metric("Marks earned so far",
                 str(round(summary["mark_so_far"], 1)) + " / 100")
        b.metric("Highest total still reachable",
                 str(round(summary["highest_possible"], 1)) + " / 100")

        if summary["highest_possible"] < settings.PASS_MARK:
            st.caption("The pass mark is " + str(settings.PASS_MARK)
                       + ". This course can no longer be passed.")

        if len(summary["alerts"]) == 0:
            st.success("No rule is triggered.")
        else:
            for alert in summary["alerts"]:
                line = "**" + alert["rule"] + "** — " + alert["message"]
                if alert["level"] == "barred":
                    st.error(line)
                elif alert["level"] == "warning":
                    st.warning(line)
                else:
                    st.info(line)

    with right:
        st.subheader("2. Model prediction")
        st.caption("For the students no rule has caught yet.")

        st.metric("Chance of not passing this course",
                  str(round(probability * 100, 1)) + " percent")

        if level == "HIGH RISK":
            st.error(level)
        elif level == "MEDIUM RISK":
            st.warning(level)
        else:
            st.success(level)

        st.write("**Why the model says this**")
        for reason in explain(values):
            st.write("- " + reason)

        with st.expander("What the model actually received"):
            st.caption("The mark sheet below, converted into the nine "
                       "indicators the model was trained on.")
            as_rows = []
            for name in FEATURES:
                as_rows.append({"indicator": name, "value": values[name]})
            st.dataframe(pd.DataFrame(as_rows), hide_index=True, width="stretch")

    # ----- saving the prediction, and what was saved before -----
    #
    # One prediction on its own says where a student stands today. A second
    # one, two weeks later, says which direction they are moving in, and
    # that is a different conversation for the advisor to have.
    verdict_box.markdown("---")
    verdict_box.markdown("### The record for this student")

    save_column, count_column = verdict_box.columns([1, 3])

    if save_column.button("Save this prediction"):
        database.save_student(student_number, program, academic_level,
                              credits_this_term)
        database.save_prediction(
            student_id=student_number,
            course_code=course_code,
            made_on=str(date.today()),
            day_of_term=day_of_term,
            mark_so_far=summary["mark_so_far"],
            highest_possible=summary["highest_possible"],
            probability=round(probability, 4),
            risk_level=level,
            policy_status=summary["status"],
        )
        save_column.success("Saved.")

    count_column.caption("The database holds "
                         + str(database.count_predictions())
                         + " saved predictions for "
                         + str(database.count_students()) + " students. "
                         + "No names, no identity numbers and no contact "
                         + "details are stored, only the university number "
                         + "and the result of the calculation.")

    history = database.get_history(student_number)
    if len(history) == 0:
        verdict_box.caption("Nothing saved for this student yet.")
    else:
        verdict_box.dataframe(history, hide_index=True, width="stretch")

    verdict_box.divider()


# ----- tab 2 : one student across all their courses -----
with tab2:
    st.subheader("One student, every course")
    st.markdown(
        "The system predicts **one course at a time** and then gathers the "
        "courses back together. This matters: a student with a good average "
        "can still be collapsing in a single course, and an average taken "
        "across the student would hide exactly the case we are trying to catch."
    )

    chosen_set = st.radio("Which set of students",
                          settings.DEMO_SETS,
                          format_func=lambda name: DEMO_LABELS[name],
                          horizontal=True)
    demo = demo_sets[chosen_set]
    st.caption(str(demo["id_student"].nunique()) + " students, "
               + str(len(demo)) + " courses between them. None of them is in "
               + "the training set, the validation set or the test set, and "
               + "no student appears in more than one of the three sets.")

    # Students carrying several courses are listed first, because they are
    # the ones that show what this screen is for.
    course_counts = demo.groupby("id_student").size().sort_values(ascending=False)

    labels = []
    id_for_label = {}
    for student_id in course_counts.index:
        count = course_counts[student_id]
        if count == 1:
            label = str(student_id) + "  (1 course)"
        else:
            label = str(student_id) + "  (" + str(count) + " courses)"
        labels.append(label)
        id_for_label[label] = student_id

    chosen_label = st.selectbox("Student number", labels)
    courses = demo[demo["id_student"] == id_for_label[chosen_label]]

    rows = []
    for position in range(len(courses)):
        course = courses.iloc[position]

        course_values = {}
        for name in FEATURES:
            course_values[name] = course[name]
        course_probability = predict(course_values)

        if course["at_risk"] == 1:
            what_happened = "did not pass"
        else:
            what_happened = "passed"

        rows.append({
            "Course": course["code_module"],
            "Term": course["code_presentation"],
            "Chance of not passing": round(course_probability * 100, 1),
            "Risk": risk_level(course_probability),
            "Marks so far (%)": round(course["early_score"], 1),
            "Work handed in (%)": round(course["submission_rate"], 1),
            "Longest gap (days)": int(course["longest_gap_days"]),
            "What happened": what_happened,
        })

    table = pd.DataFrame(rows).sort_values("Chance of not passing", ascending=False)
    worst = table["Chance of not passing"].max()
    worst_level = risk_level(worst / 100)

    if worst_level == "HIGH RISK":
        st.error("**This student needs contacting.** The worst course is at "
                 + str(worst) + " percent.")
    elif worst_level == "MEDIUM RISK":
        st.warning("**Keep this student under review.** The worst course is at "
                   + str(worst) + " percent.")
    else:
        st.success("**No course is a concern.** The worst is at "
                   + str(worst) + " percent.")

    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("This student was kept out of the training set, the validation "
               "set and the test set, so the model has never seen them and "
               "their result is not part of any figure on the next tab. The "
               "last column is what really happened, shown so the prediction "
               "can be judged honestly rather than taken on trust.")


# ----- tab 3 : the results -----
with tab3:
    st.subheader("Which dataset the system was built on")
    show_table("results_dataset_comparison.csv",
               "The same model, same settings and same seed, trained on three "
               "universities. Valencia has attendance taken from campus wifi "
               "and the previous year's performance, but no coursework mark "
               "before the exam, and its recall collapses to 0.30. The other "
               "two both have such a mark and both reach 0.69. That is the "
               "evidence for predicting after the midterm.")

    st.subheader("How the four algorithms compared")
    show_table("results_model_comparison.csv", "")

    st.subheader("Choosing the alert threshold")
    show_table("results_threshold_table.csv",
               "We chose " + str(settings.ALERT_THRESHOLD) + ". Missing a "
               "student costs that student a term. A false alarm costs the "
               "advisor a phone call. The two are not worth the same, so the "
               "threshold is not left at the usual 0.50.")

    st.subheader("Is it fair across groups")
    show_table("results_fairness.csv",
               "None of these three attributes is given to the model. They are "
               "used only to audit it afterwards. Recall holds between 0.771 "
               "and 0.829 across every group. Precision does not: 0.596 for "
               "female students against 0.696 for male students, which means "
               "proportionally more unnecessary calls to one group. We report "
               "it rather than hide it.")

    st.subheader("Which indicators matter most")
    try:
        importance = pd.read_csv(
            settings.project_file("results_feature_importance.csv"))
        chart_data = importance.set_index("Feature")
        st.bar_chart(chart_data["Share_%"])
        st.caption("Behaviour on the platform carries more of the decision "
                   "than the early mark does.")
    except Exception:
        st.info("results_feature_importance.csv not found")


# ----- tab 4 -----
with tab4:
    st.markdown("""
**What this system does**

It looks at one student in one course at the middle of the term and estimates
whether that student is heading towards not passing, while there is still
time to do something about it. The academic advisor decides what to do.

**Why there are two halves**

Some things are rules, not predictions. Four unexcused absences means the
student is barred from the course. Needing more than the remaining marks to
reach 60 means the course is already lost. Neither is a probability and
neither needs a model, so the policy engine simply counts them.

The model is for the rest: the student no rule has caught yet, whose marks
are still passable, but whose pattern looks like the students who did not
pass.

**Where the attendance rule and the cumulative average sit**

In the policy engine, and not yet in the model. Not because they were
rejected: no dataset we could obtain carries either column alongside the
rest, so the model was never taught to read them, and handing a trained
model a number it has never seen produces noise rather than a prediction.
Both are collected on this screen and both are applied as rules, which is
how the university applies them anyway. The design keeps a named slot for
each, and switching them on is four steps and a retrain.

**Data**

Open University Learning Analytics Dataset, Kuzilek, Hlosta and Zdrahal,
Scientific Data 4, 170171, 2017, licence CC-BY 4.0. Two further datasets,
from UPV Valencia and from the University of KwaZulu-Natal, were used to
test the same model and are reported on the Model performance tab.

**What is still an assumption**

The credit load conversion: we treat a full load here as the same pressure
as a full load at the Open University. Reasonable, not proven, and the
first thing to check against real data from our own university.

**The limit nobody should forget**

This has never been tested on data from our own university, because we were
not given any.
    """)
