"""
Smart Academic Risk Detection System - the advisor interface.

Three screens: Scan reads a cohort file and flags students at risk,
Register lists one row per student, and Evidence shows how the system
performs.

The uploaded file holds one row per event. The system drops events after
the cut day, builds the six indicators, and scores every student. An
indicator with no data is left empty, not set to zero.
"""

import base64
import datetime
import io

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

import settings
import database
import indicators as build

INDICATORS = settings.INDICATORS

# Indicator names in both languages, for display.
WORDS = {
    "score_mean":           ("متوسط الدرجات حتى الآن", "mean mark so far"),
    "weight_sum":           ("وزن ما سُلِّم من الأعمال",
                             "weight of work handed in"),
    "visit_day_last":       ("يوم آخر دخول للمنصة", "day of the last visit"),
    "late_days_mean":       ("متوسط أيام التأخير", "mean days late"),
    "clicks_homepage":      ("نقرات صفحة المقرر",
                             "clicks on the course page"),
    "highest_education":    ("أعلى مؤهل دراسي", "highest qualification"),
    "id_student":           ("رقم الطالب", "student number"),
    "code_module":          ("رمز المقرر", "course code"),
    "code_presentation":    ("الفصل الدراسي", "term"),
    "event_type":           ("نوع السجل", "kind of record"),
    "date_due":             ("يوم استحقاق العمل", "day the work was due"),
    "date_submitted":       ("يوم التسليم", "day it was handed in"),
    "score":                ("الدرجة", "the mark"),
    "weight":               ("وزن العمل", "weight of the work"),
    "date":                 ("يوم الدخول", "day of the visit"),
    "clicks":               ("عدد النقرات", "number of clicks"),
}

# Arabic names for the qualification values. A value not listed here is
# shown as it is.
QUALIFICATIONS = {
    "No Formal quals":             "بدون مؤهل رسمي",
    "Lower Than A Level":          "أقل من الثانوية",
    "A Level or Equivalent":       "الثانوية أو ما يعادلها",
    "HE Qualification":            "مؤهل جامعي",
    "Post Graduate Qualification": "مؤهل دراسات عليا",
}

# The unit each indicator is measured in.
UNITS = {
    "score_mean":      ("من 100", "out of 100"),
    "weight_sum":      ("من 100", "out of 100"),
    "visit_day_last":  ("يوم", "day"),
    "late_days_mean":  ("يوم", "days"),
    "clicks_homepage": ("نقرة", "clicks"),
}

TEXT = {
    "title":        ("نظام الكشف المبكر عن التعثر الأكاديمي",
                     "Smart Academic Risk Detection System"),
    "subtitle":     ("يقرأ سجلات الدفعة، ويبني المؤشرات، ثم ينذر المرشد",
                     "Reads the cohort's records, builds the indicators, "
                     "notifies the advisor"),
    "tab_scan":     ("فحص الدفعة", "Scan"),
    "tab_register": ("سجل الطلاب", "Register"),
    "tab_evidence": ("أداء النظام", "System performance"),
    "source":       ("مصدر البيانات", "Where the data comes from"),
    "src_file":     ("ملف أرفعه", "A file I upload"),
    "src_db":       ("قاعدة بيانات الجامعة", "The university database"),
    "db_closed":    ("غير متاح في هذه النسخة. النظام مصمم ليقرأ من قاعدة "
                     "الجامعة مباشرة، ولم يُمنح الوصول بعد.",
                     "Not available in this version. The system is built to "
                     "read the university database directly and has not been "
                     "given access yet."),
    "upload":       ("ارفع ملف الدفعة", "Upload a cohort file"),
    "no_file":      ("لم يُرفع ملف بعد.", "No file yet."),
    "run":          ("ابدأ الفحص", "Run the scan"),
    "scanning":     ("جاري الفحص", "Scanning"),
    "why_stopped":  ("لا يُنذَر عن طالب لا يُعرف رقمه، ولا يُبنى إنذار على "
                     "ملف لا يُبنى منه مؤشر واحد. أضف ما "
                     "ينقص أدناه وارفع الملف مرة أخرى.",
                     "The system does not warn about a student whose number "
                     "it does not know, and does not build an alert on a "
                     "file that builds not a single indicator. Add what is "
                     "listed below and upload the "
                     "file again."),
    "gap_text":     ("خانات غير قابلة للقراءة كرقم",
                     "Cells that could not be read as a number"),
    "gap_text_note": ("قُرئت كقيمة مجهولة، لا كصفر — كما يُقرأ أي نقص آخر.",
                      "They were read as not known, not as a zero, the "
                      "same as any other gap."),
    "students":     ("طلاب", "students"),
    "flagged":      ("منذَر عنهم", "flagged"),
    "clear":        ("لا شيء عليهم", "clear"),
    "no_value":     ("لا توجد قيمة", "no value"),
    "flagged_block": ("الطلاب المنذَر عنهم — ومؤشرات كل واحد",
                      "The students flagged, and what was read for each"),
    "student":      ("الطالب", "Student"),
    "prob":         ("احتمال التعثر", "Probability"),
    "state":        ("حالة الإنذار", "Notice"),
    "waiting":      ("في الانتظار", "waiting"),
    "sent":         ("أُرسل للمرشد", "sent to the advisor"),
    "send":         ("أرسل للمرشد الأكاديمي", "Notify the academic advisor"),
    "sent_at":      ("أُرسل في", "Sent"),
    "stop_note":    ("ينتهي دور النظام هنا. ما يفعله المرشد بعد ذلك ليس من "
                     "شأن النظام ولا يُسجَّل فيه.",
                     "The system's part ends here. What the advisor does next "
                     "is not the system's business and is not recorded in it."),
    "ev_what":      ("هذه الأرقام تصف النموذج نفسه، لا الملف الذي تفحصه، "
                     "ولا تتغير بتغيّر الدفعة — لأن الفصل الجاري لم تُعرف "
                     "نتيجته بعد.\n\n"
                     "والصفحة تُقرأ من أعلى لأسفل: كيف اختِيرت المؤشرات، "
                     "ثم ماذا يقرأ النموذج، ثم لماذا هذه الخوارزمية وهذا "
                     "الحد، ثم هل يعامل الجميع بالمثل — وكلها قيست على "
                     "مجموعة التحقق، وهي التي اتُّخذ عليها كل قرار. وآخر "
                     "قسم وحده هو النتيجة النهائية: شريحة محجوزة فُتحت "
                     "مرة واحدة بعد تثبيت كل شيء.",
                     "These numbers describe the model itself, not the "
                     "file you are scanning, and they do not change when "
                     "the cohort changes - because a term still running "
                     "has no outcome yet.\n\n"
                     "The page reads from the top down: how the "
                     "indicators were chosen, then what the model reads, "
                     "then why this algorithm and this threshold, then "
                     "whether it treats everyone the same. All of those "
                     "were measured on the validation slice, which is "
                     "where every decision was made. The last section "
                     "alone is the final result: a held-out slice, "
                     "opened once, after everything was fixed."),
    "ev_test":      ("الاختبار النهائي — فُتح مرة واحدة",
                     "The final test — opened once"),
    # The seven sections of the performance page, in display order.
    "ev_chosen":    ("من أين جاءت المؤشرات الستة",
                     "Where the six indicators came from"),
    "ev_chosen_note": ("عُرض كل عمود في البيانات الخام — واحد وسبعون "
                       "مرشحاً — على تجربة اختيار. الجدول يبيّن كم أضاف "
                       "كل مؤشر، وأي مرشح لم يُثبت نفسه على بيانات لم "
                       "تُستخدم في اختياره.",
                       "Every column in the raw data - seventy-one "
                       "candidates - was offered to a selection "
                       "experiment. The table shows what each indicator "
                       "added, and which candidate did not hold up on "
                       "data its selection never saw."),
    "ev_chosen_table": ("كل مرشح، وما أضافه، وهل بقي",
                        "Every candidate, what it added, whether it stayed"),
    "chart_chosen": ("ما أضافه كل مؤشر عند اختياره",
                     "What each indicator added when it was chosen"),
    "col_added":    ("ما أضافه", "What it added"),

    "ev_identity":  ("ولا عمود عن هوية الطالب",
                     "Not one column about who the student is"),
    "ev_identity_note": ("هذه الأعمدة عُرضت على التجربة نفسها، ولم تختر "
                         "أياً منها. لم تُستبعد — التجربة لم تجد فيها "
                         "ما يضيف.",
                         "These columns were offered to the same "
                         "experiment and it chose none of them. They were "
                         "not excluded; the experiment found nothing "
                         "in them to add."),
    "id_gender":    ("الجنس", "gender"),
    "id_region":    ("المنطقة", "region"),
    "id_imd":       ("المستوى الاجتماعي", "deprivation band"),
    "id_age":       ("الفئة العمرية", "age band"),
    "id_disability": ("الإعاقة", "disability"),
    "ev_identity_one": ("العمود الأصلي الوحيد الذي اختارته التجربة هو "
                        "أعلى مؤهل دراسي، وهو خلفية أكاديمية لا صفة "
                        "شخصية.",
                        "The one original column the experiment did "
                        "choose is the highest qualification, which is "
                        "academic background rather than a personal "
                        "characteristic."),

    "ev_models_head": ("لماذا هذه الخوارزمية", "Why this algorithm"),
    "ev_models_note": ("أربع خوارزميات على نفس البيانات ونفس المؤشرات "
                       "وإعدادات عادية غير مضبوطة، حتى تكون المقارنة بين "
                       "الخوارزميات لا بين الوقت المبذول على كل واحدة.",
                       "Four algorithms on the same data, the same "
                       "indicators and ordinary untuned settings, so the "
                       "comparison is between the algorithms and not "
                       "between how much time was spent on each."),

    "ev_thresh_head": ("لماذا الحد 0.20", "Why the threshold is 0.20"),
    "ev_thresh_note": ("الجدول كامل، لا الصف المختار وحده. كل ما شُدّ "
                       "الحد ارتفعت الدقة وفات متعثرون أكثر — والاختيار "
                       "هو أي الخطأين أغلى.",
                       "The whole table, not the chosen row alone. "
                       "Tightening the threshold raises precision and "
                       "misses more students at risk; the choice is "
                       "which of the two mistakes costs more."),

    "ev_fair_head": ("هل يعامل الجميع بالمثل",
                     "Does it treat everyone the same"),
    "ev_test_head": ("النتيجة النهائية", "The final result"),
    "ev_test_note": ("شريحة محجوزة لم تُفتح إلا مرة واحدة، بعد تثبيت "
                     "المؤشرات والخوارزمية والحد. لم يُعدَّل بعدها شيء.",
                     "A held-out slice, opened once, after the "
                     "indicators, the algorithm and the threshold were "
                     "all fixed. Nothing was adjusted afterwards."),
    "ev_no_file":   ("ملف النتائج غير موجود بجانب التطبيق",
                     "The result file is not beside the application"),

    "ev_thresh":    ("كل حد فاصل جُرِّب — مجموعة التحقق، لأن عليها اختِير الحد",
                     "Every threshold tried — validation set, because that "
                     "is where the threshold was chosen"),
    "ev_models":    ("الخوارزميات الأربع عند حد ٠٫٢٠ — مجموعة التحقق. "
                     "الأرقام هنا لا تقارن بالاختبار النهائي أدناه، لأنها "
                     "على شريحة أخرى.",
                     "The four algorithms at a threshold of 0.20 - the "
                     "validation slice. These numbers are not comparable "
                     "with the final test below: a different slice."),
    "ev_fair":      ("فحص العدالة — على مجموعة التحقق",
                     "Fairness audit — on the validation set"),
    "ev_fair_note": ("هذه الأعمدة لا تُعطى للنموذج إطلاقاً. تُقرأ بعد أن "
                     "ينتهي، لمراجعته. والفحص على مجموعة التحقق لا على "
                     "الاختبار، فمجموع كل خاصية ٢٢٤٣ — عدا المستوى "
                     "الاجتماعي، فمجموعه ٢١٤٥، لأن ٩٨ طالباً لا تحمل "
                     "سجلاتهم قيمة له، ولا تُفترض لهم قيمة.",
                     "These columns are never given to the model. They "
                     "are read afterwards, to audit it. The audit runs "
                     "on the validation slice and not on the test slice, "
                     "so each attribute totals 2,243 - except the deprivation band, "
                     "which totals 2,145, because 98 students carry no "
                     "value for it and none is assumed for them."),
    "ev_weight":    ("وزن كل مؤشر في القرار", "What each indicator carries"),
    "ev_small":     ("الفئة التي عدد طلابها أقل من خمسين تُعدّ ولا تُقاس: "
                     "خليّة فارغة تعني أنها لم تُقَس، لا أنها صفر.",
                     "A group of fewer than fifty students is counted but "
                     "not scored: an empty cell means it was not measured, "
                     "not that it is zero."),
    "lang":         ("English", "العربية"),
    "src_file_sub": ("ملف الدفعة بصيغة مفصولة بفواصل",
                     "A cohort file in comma separated form"),
    "src_db_sub":   ("قراءة مباشرة من سجلات الجامعة",
                     "Read straight from the university's records"),
    "ready":        ("متاح", "available"),
    "not_ready":    ("غير متاح", "not available"),
    # The three stages of a scan, as shown to the advisor.
    "step1":        ("تهيئة البيانات", "Prepare the data"),
    "step2_":       ("مسح البيانات", "Scan"),
    "step3_":       ("النتيجة", "Result"),
    "of_cohort":    ("من الدفعة", "of the cohort"),
    "f_file":       ("الملف المفحوص", "The file scanned"),
    "f_records":    ("سجلات المقررات", "Course records"),
    "f_cut":        ("يوم القطع", "Cut day"),
    "f_threshold":  ("حد الإنذار", "Alert threshold"),
    "gap_some":     ("{have} من {total} مؤشرات متوفرة في هذا الملف",
                     "{have} of {total} indicators are available in this "
                     "file"),
    "gap_some_note": ("ما لا يوفره الملف يُترك فارغاً ويُقرأ كغير معروف، "
                      "والفحص يمضي بما توفر.",
                      "What the file does not supply is left empty and "
                      "read as not known; the scan goes on with what "
                      "there is."),
    "gap_none":     ("لا يتوفر أي مؤشر في هذا الملف",
                     "No indicator is available in this file"),
    "gap_no_keys":  ("الملف لا يحمل أعمدة لا يقوم الفحص بدونها",
                     "The file is missing columns the scan cannot run "
                     "without"),
    "search":       ("ابحث برقم الطالب", "Search by student number"),
    "back":         ("سجل الطلاب", "Register"),
    "nobody":       ("لا يوجد طالب بهذا الرقم", "No student with that number"),
    "click_row":    ("انقر صفاً لفتح سجل الطالب",
                     "Click a row to open that student's record"),
    "file_known":   ("هذا الملف موجود عندنا — رُفع في {when} ويحمل {rows} سجلاً. "
                     "الفحص سيضيف عملية جديدة عليه، ولن يُخزَّن الملف مرتين.",
                     "This file is already stored — uploaded {when}, "
                     "carrying {rows} records. Scanning adds another run on "
                     "it, and the file is not stored twice."),
    "file_updated": ("ملف بهذا الاسم موجود ({n} نسخة)، لكن محتوى الملف الذي "
                     "رفعته الآن مختلف. سيُحفظ كنسخة محدَّثة مستقلة.",
                     "A file of this name is already stored ({n} version(s)), "
                     "but what you just uploaded is not the same file. It "
                     "will be kept as a separate, updated version."),
    "stored":       ("الملفات المخزَّنة وعمليات فحصها",
                     "Stored files and their scans"),
    "no_stored":    ("لم يُفحص أي ملف بعد.", "Nothing has been scanned yet."),
    "all_clear":    ("لا أحد على هذا الملف يستدعي إنذاراً: لم يتجاوز أي "
                     "طالب الحد.",
                     "Nobody on this file needs an alert: no student "
                     "passed the threshold."),
    "run_again":    ("افحص مرة أخرى", "Scan again"),
    "run_again_note": ("هذا الملف مفحوص من قبل. الفحص مرة أخرى يُسجَّل "
                       "فحصاً جديداً بتاريخه، ولا يُلغي الأول.",
                       "This file has been scanned before. Scanning it "
                       "again is stored as a new scan with its own date, "
                       "and does not replace the first."),
    "open_run":     ("افتح هذه النتيجة", "Open this result"),
    "rescan":       ("افحص الملف من جديد", "Scan this file again"),
    "from_store":   ("هذه نتيجة محفوظة من {when} — مقروءة كما كُتبت يومها، "
                     "دون إعادة فحص.",
                     "A stored result from {when}, read back as it was "
                     "written on the day, without scanning again."),
    "drag":         ("اضغط الزر لاختيار ملف من جهازك، أو اسحب الملف وأفلته "
                     "داخل الإطار.",
                     "Use the button to pick a file from your computer, or "
                     "drag a file and drop it inside the box."),
    "bad_encoding": ("تعذّرت قراءة هذا الملف. احفظه من جديد بترميز UTF-8 "
                     "ثم ارفعه مرة أخرى. في إكسل: حفظ باسم ← "
                     "CSV UTF-8.",
                     "This file could not be read. Save it again as UTF-8 "
                     "and upload it once more. In Excel: Save As, then "
                     "CSV UTF-8."),
    "bad_file":     ("هذا الملف ليس جدول دفعة يمكن قراءته. المطلوب ملف "
                     "مفصول بفواصل، سطره الأول أسماء الأعمدة، وفيه عمود "
                     "لرقم الطالب ورمز المقرر.",
                     "This file is not a cohort table the system can read. "
                     "It needs to be comma separated, with the column names "
                     "on the first line, and to carry a student number and "
                     "a course code."),
    "empty_file":   ("الملف لا يحتوي على أي سجل.",
                     "This file carries no records at all."),

    # Column headings.
    "col_file":     ("الملف", "File"),
    "col_when":     ("وقت الفحص", "Scanned at"),
    "col_students": ("عدد الطلاب", "Students"),
    "col_flagged":  ("منذَر عنهم", "Flagged"),
    "col_carried":  ("مؤشرات الملف", "Indicators in the file"),
    "col_failing":  ("مقررات متعثرة", "Courses in trouble"),
    "col_courses":  ("مقررات مسجَّلة", "Courses taken"),
    "col_course":   ("المقرر", "Course"),
    "col_reason":   ("السبب", "Reason"),
    "col_prob":     ("احتمال التعثر", "Probability"),
    "col_read":     ("مؤشرات مقروءة", "Indicators read"),
    "col_indicator": ("المؤشر", "Indicator"),
    "col_value":    ("قيمته على هذا السجل", "its value on this record"),
    "col_share":    ("نصيبه من القرار", "share of the decision"),
    "col_group":    ("الفئة", "Group"),
    "col_threshold": ("الحد الفاصل", "Threshold"),
    "m_recall":     ("الاستدعاء", "Recall"),
    "m_precision":  ("الدقة", "Precision"),
    "grp_gender_F": ("إناث", "female"),
    "grp_gender_M": ("ذكور", "male"),
    "grp_age_band_0-35":  ("حتى ٣٥ سنة", "up to 35"),
    "grp_age_band_35-55": ("٣٥ إلى ٥٥ سنة", "35 to 55"),
    "grp_age_band_55<=":  ("٥٥ فأكثر", "55 and over"),
    "grp_disability_True":  ("بإعاقة مسجَّلة", "disability recorded"),
    "grp_disability_False": ("بلا إعاقة مسجَّلة", "no disability recorded"),
    "against_threshold": ("مقابل حد الإنذار {t}", "against the alert threshold of {t}"),
    "push_note":    ("أقوى {n} مؤشرات دفعت نحو الإنذار، وتحمل {total}٪ من "
                     "القرار. الباقي موزَّع على بقية المؤشرات.",
                     "The {n} indicators that pushed hardest towards an "
                     "alert, carrying {total}% of the decision between "
                     "them. The rest is spread across the others."),

    # Captions for the three charts on the performance screen.
    "chart_threshold": ("كيف يتبادل الاستدعاء والدقة عند كل حد فاصل. "
                        "اختِير ٠٫٢٠ لأن الطالب المفقود يخسر فصلاً، "
                        "والإنذار الكاذب يكلّف مكالمة، وما يُفعل بالإنذار "
                        "قرار المرشد لا قرار النظام.",
                        "How recall and precision trade off at each "
                        "threshold. 0.20 was chosen because a missed "
                        "student loses a term, and a false alarm costs a "
                        "phone call."),
    "chart_fair":      ("الاستدعاء والدقة لكل فئة. الفجوة في الدقة بين "
                        "الإناث والذكور معروضة كما خرجت، ولم تُخفَ.",
                        "Recall and precision for each group. The "
                        "precision gap between female and male students "
                        "is shown as it came out, not hidden."),
    "chart_weight":    ("نصيب كل مؤشر من القرار. الدرجة المبكرة وحدها "
                        "تحمل قرابة نصفه، والخمسة الباقية تحمل الباقي.",
                        "What each indicator carries of the decision. The "
                        "early mark alone carries close to half of it, and "
                        "the other five carry the rest."),
}

def count_students(n):
    """A count of students, with the right Arabic plural form.

    Arabic has four forms: one, two, three to ten, and eleven upwards.
    """
    if st.session_state["lang"] != "ar":
        return str(n) + (" student" if n == 1 else " students")
    if n == 1:
        return "طالب واحد"
    if n == 2:
        return "طالبان"
    if 3 <= n <= 10:
        return str(n) + " طلاب"
    return str(n) + " طالباً"

def count_cells(n):
    """A count of cells, agreeing in Arabic."""
    if st.session_state["lang"] != "ar":
        return str(n) + (" cell" if n == 1 else " cells")
    if n == 1:
        return "خانة واحدة"
    if n == 2:
        return "خانتان"
    if 3 <= n <= 10:
        return str(n) + " خانات"
    return str(n) + " خانة"


def count_records(n):
    """The same, for course records rather than students."""
    if st.session_state["lang"] != "ar":
        return str(n) + (" course record" if n == 1 else " course records")
    if n == 1:
        return "سجل مقرر واحد"
    if n == 2:
        return "سجلّا مقرر"
    if 3 <= n <= 10:
        return str(n) + " سجلات مقرر"
    return str(n) + " سجل مقرر"

def count_indicators(seen, total):
    """How many indicators were readable, out of how many there are."""
    if st.session_state["lang"] != "ar":
        return ("The system read " + str(seen) + " of " + str(total)
                + " indicators for this record")
    word_seen = str(seen) + (" مؤشرات" if 3 <= seen <= 10 else " مؤشراً")
    if seen == 1:
        word_seen = "مؤشراً واحداً"
    return "قرأ النظام " + word_seen + " من " + str(total) + " لهذا السجل"

# Page styling: colours, font and shapes.
STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --green:#1b8354; --green-d:#074d31; --green-l:#eaf4ef; --green-br:#6fe3ab;
  --ink:#212529; --ink-2:#4b5563; --muted:#6b7280;
  --paper:#fff; --band:#f3f4f6; --line:#e5e7eb;
  --red:#b3261e; --red-l:#fdecea;
  --orange:#b45309; --orange-l:#fff4e8;
  --font:"IBM Plex Sans Arabic","Segoe UI",system-ui,sans-serif;
}
/* The same palette in dark. Only the surfaces and the text change: the
   university green is the brand and stays as it is. Every rule below
   reads these names, so nothing else has to be written twice. */
@media (prefers-color-scheme: dark){
  :root{
    --ink:#e8eaed; --ink-2:#c2c8cf; --muted:#9aa1a9;
    --paper:#1b1f24; --band:#121417; --line:#2e343c;
    --green-l:#15362a;
    --red:#ff8a80; --red-l:#3a1f1d;
    --orange:#f0a35e; --orange-l:#3a2a18;
  }
}
/* The font is inherited rather than forced: Streamlit's icon font has to
   stay on the elements that set it. */
html, body, .stApp { font-family: var(--font); }
button, input, select, textarea { font-family: var(--font); }
.stApp [data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded'!important}
.stApp { background: var(--band); color: var(--ink); }
.block-container { padding-top: 1.2rem; max-width: 1280px; }
#MainMenu, footer, header { visibility: hidden; }
/* Streamlit's own widgets are brought onto the palette above, so they
   read the same in either mode. The uploader's button is named on its
   own: a blanket rule would colour its label and leave its background to
   Streamlit, which is how a button ends up dark text on dark. */
.stApp [data-testid="stTab"], .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stRadioOption"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stFileUploaderDropzoneInstructions"],
.stApp [data-testid="stFileUploaderDropzoneInstructions"] *{color:var(--ink)}
.stApp [data-testid="stFileUploaderDropzone"]{background:var(--paper)}
.stApp [data-testid="stFileUploaderDropzone"] button{
  background:var(--green)!important;color:#fff!important;border:0;
  border-radius:8px;font-weight:600}
.stApp [data-testid="stFileUploaderDropzone"] button *{color:#fff!important}
.stApp [data-testid="stFileUploaderDropzone"] button:hover{
  background:var(--green-d)!important}
/* the band across the top */
.hero{background:var(--green-d);color:#fff;border-radius:14px;padding:34px 26px;
      text-align:center;position:relative;overflow:hidden;margin-bottom:22px}
.hero:after{content:"";position:absolute;inset-inline-start:-110px;top:-170px;
            width:460px;height:460px;border-radius:50%;background:rgba(27,131,84,.5)}
.hero .kick{font-size:13.5px;color:#9fd3b9;letter-spacing:.3px;margin-bottom:10px;
            position:relative;z-index:1}
.hero h1{font-size:clamp(24px,4vw,36px);font-weight:700;line-height:1.3;
         position:relative;z-index:1;margin:0}
.hero .lede{font-size:15.5px;color:#cfe6da;margin-top:12px;position:relative;z-index:1}
/* the counting tiles */
.split{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
       gap:14px;margin:4px 0 20px}
.tile{background:var(--paper);border:1px solid var(--line);border-radius:12px;
      padding:22px 16px;text-align:center;min-width:0;height:100%;
      display:flex;flex-direction:column;justify-content:center}
/* the three tiles: equal height, with a slot for a second line */
.tile .n{font-size:36px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.tile .t{font-size:14.5px;color:var(--ink-2);margin-top:8px}
.tile.r .n{color:var(--red)} .tile.o .n{color:var(--orange)} .tile.g .n{color:var(--green)}
/* the flagged tile: a Streamlit button styled as a tile */
.st-key-to_register{width:100%}
.st-key-to_register .stButton{display:block;width:100%}
.st-key-to_register .stButton>button{width:100%!important;
  min-height:136px;background:var(--paper);
  border:1px solid var(--line);border-radius:12px;padding:22px 16px;
  text-align:center;box-shadow:none;transition:border-color .15s,
  box-shadow .15s}
.st-key-to_register .stButton>button:hover{background:var(--paper);
  border-color:var(--orange);box-shadow:0 2px 10px rgba(180,83,9,.12)}
.st-key-to_register .stButton>button:focus{box-shadow:none;
  border-color:var(--orange)}
.st-key-to_register .stButton>button p{margin:0;line-height:1.35;
  color:var(--ink-2);font-size:14.5px;font-weight:400}
.st-key-to_register .stButton>button p strong{display:block;
  font-size:36px;font-weight:700;line-height:1;color:var(--orange);
  font-variant-numeric:tabular-nums;margin-bottom:8px}
.st-key-to_register .stButton>button p em{display:block;
  font-style:normal;font-size:13px;color:var(--ink-2);margin-top:4px}
.tile .u{font-size:12px;color:var(--muted);margin-top:4px}
.box.warn{border-inline-start:3px solid var(--orange);
  background:var(--orange-l)}
.box.stopped{border-inline-start:3px solid var(--red);
  background:var(--red-l)}
.box .gaphead{font-weight:700;margin-bottom:6px}
.box .gaplist{color:var(--ink-2);font-size:14px;line-height:1.95}
/* the register */
.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;
      overflow:hidden;margin-bottom:20px}
.card .scroll{overflow-x:auto}
.card table{width:100%;border-collapse:collapse;font-size:15px}
.card thead th{text-align:start;font-weight:500;font-size:13px;color:var(--muted);
               padding:12px 18px;border-bottom:1px solid var(--line);white-space:nowrap}
.card tbody td{padding:14px 18px;border-bottom:1px solid var(--band)}
.card tbody tr:last-child td{border-bottom:0}
.card tr.modelrow td:first-child{box-shadow:inset 3px 0 0 var(--orange)}
.blockhead{background:var(--band);padding:12px 18px;font-size:13.5px;
           font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--line)}
.blockhead .sm{font-weight:400;color:var(--muted);font-size:12.5px}
.sid{font-weight:700;font-variant-numeric:tabular-nums;font-size:16px}
.tag{display:inline-block;border-radius:20px;padding:4px 12px;font-size:12.5px;
     font-weight:600;white-space:nowrap}
.tag.r{background:var(--red-l);color:var(--red)}
.tag.o{background:var(--orange-l);color:var(--orange)}
.tag.g{background:var(--green-l);color:var(--green)}
/* one student, opened */
.box{background:var(--paper);border:1px solid var(--line);border-radius:12px;
     padding:24px;margin-bottom:16px}
.box h3{font-size:18px;font-weight:700;margin:0 0 2px}
.box .cap{font-size:12.5px;color:var(--muted);margin-bottom:16px}
.why{font-size:14.5px;color:var(--ink-2);line-height:1.75;margin:0}
.push{display:flex;justify-content:space-between;gap:14px;padding:10px 0;
      border-bottom:1px solid var(--band);font-size:14.5px}
.push:last-child{border-bottom:0}
.push.hd{font-size:12px;font-weight:600;color:var(--muted);padding:6px 0;
         letter-spacing:.2px}
.push.hd .v,.push.hd .s{font-weight:600;color:var(--muted)}
.push .v{font-variant-numeric:tabular-nums;color:var(--ink-2)}
.push .s{font-weight:700;font-variant-numeric:tabular-nums;color:var(--orange)}
.note{font-size:13px;color:var(--muted);line-height:1.7;margin-top:12px}
/* one indicator and its share of the decision, with the name on its own line */
.wrow{display:flex;align-items:center;gap:14px;padding:9px 0;
      border-bottom:1px solid var(--band);font-size:14px}
.wrow:last-child{border-bottom:0}
.wrow .nm{flex:0 0 220px}
.wrow .track{flex:1 1 auto;height:9px;border-radius:5px;background:var(--band)}
.wrow .fill{display:block;height:9px;border-radius:5px;background:var(--green)}
.wrow .s{flex:0 0 72px;text-align:end;font-weight:700;color:var(--green);
         font-variant-numeric:tabular-nums}
/* the bar across the very top, with the university's mark on it */
/* the mark stays on the right in both languages */
.nav{background:var(--paper);border:1px solid var(--line);border-radius:12px;
     padding:12px 18px;display:flex;align-items:center;margin-bottom:14px;
     direction:rtl}
.nav .logo img{height:34px;width:auto;display:block}
/* the steps of the scan */
.steps{display:flex;background:var(--paper);border:1px solid var(--line);
       border-radius:12px;overflow:hidden;margin-bottom:16px;flex-wrap:wrap}
.sp2{flex:1 1 160px;padding:13px 15px;display:flex;align-items:center;gap:10px;
     border-inline-start:1px solid var(--band);min-width:0}
.sp2:first-child{border-inline-start:0}
.sp2 .dot{width:24px;height:24px;border-radius:50%;background:var(--band);
          color:var(--muted);font-size:12px;font-weight:700;display:flex;
          align-items:center;justify-content:center;flex:0 0 24px}
.sp2.done .dot{background:var(--green);color:#fff}
.sp2 .tx{font-size:13px;color:var(--muted);min-width:0}
.sp2.done .tx{color:var(--ink);font-weight:600}
/* the breadcrumb on a student's page */
.crumb{font-size:13px;color:var(--muted);margin-bottom:10px}
.crumb b{color:var(--ink-2);font-weight:600}
/* streamlit's own pieces, brought into line */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-size:15px;color:var(--ink-2);padding:8px 14px}
.stTabs [aria-selected="true"]{color:var(--green)!important;font-weight:600}
.stButton button{background:var(--green);color:#fff;border:0;border-radius:8px;
                 padding:11px 26px;font-size:15px;font-weight:600}
.stButton button:hover{background:var(--green-d);color:#fff}
.stButton button p{font-weight:600}
/* Streamlit's tab underline and radio dot, set to the project green. */
.stApp .react-aria-SelectionIndicator{background:var(--green)!important}
.stApp [data-testid="stRadioOption"]:has(input:checked) > div > div:first-child{
  background:var(--green)!important}
/* The hero line is centred on the band. A rule that names the heading
   itself is needed: the centring on the band is inherited, and an
   inherited value never wins against one aimed at the element. */
.stApp .hero h1{text-align:center}
/* These sit at the end because a rule only overrides what comes before
   it. The palette above is set on :root and reaches every rule by name;
   these four name elements, so they have to follow those elements. */
@media (prefers-color-scheme: dark){
  /* The green reads as a fill on white but not as words on a dark
     surface, so where it carries text or a thin bar it steps up. */
  .stTabs [aria-selected="true"]{color:var(--green-br)!important}
  .wrow .fill{background:var(--green-br)}
  .wrow .s{color:var(--green-br)}
  /* The university's mark is artwork drawn for a white ground and is not
     ours to recolour, so the bar it sits on keeps its own. */
  .nav{background:#fff;border-color:#e5e7eb}
}
</style>
"""

# Values inside the result tables, in both languages. The tables and the
# charts both read this, so a value reads the same in each.
CELLS = {
    "random (final test)": ("الشريحة المحجوزة — فُتحت مرة واحدة",
                            "the held-out slice, opened once"),
    # the verdict column of the selection table
    "keeps its place": ("بقي", "stayed"),
    "not confirmed": ("لم يُثبت نفسه", "did not hold up"),
    # The three columns the fairness audit reads, and their values.
    "gender": ("الجنس", "gender"),
    "region": ("المنطقة", "region"),
    "imd_band": ("المستوى الاجتماعي", "deprivation band"),
    # The regions as the data publishes them, with Arabic names.
    "East Anglian Region": ("شرق أنجليا", "East Anglian Region"),
    "East Midlands Region": ("شرق الميدلاندز", "East Midlands Region"),
    "Ireland": ("أيرلندا", "Ireland"),
    "London Region": ("لندن", "London Region"),
    "North Region": ("الشمال", "North Region"),
    "North Western Region": ("الشمال الغربي", "North Western Region"),
    "Scotland": ("إسكتلندا", "Scotland"),
    "South East Region": ("الجنوب الشرقي", "South East Region"),
    "South Region": ("الجنوب", "South Region"),
    "South West Region": ("الجنوب الغربي", "South West Region"),
    "Wales": ("ويلز", "Wales"),
    "West Midlands Region": ("غرب الميدلاندز", "West Midlands Region"),
    "Yorkshire Region": ("يوركشاير", "Yorkshire Region"),
    "age_band": ("الفئة العمرية", "age band"),
    "disability": ("إعاقة مسجَّلة", "disability recorded"),
    "F": ("إناث", "female"),
    "M": ("ذكور", "male"),
    "0-35": ("حتى ٣٥ سنة", "up to 35"),
    "35-55": ("٣٥ إلى ٥٥ سنة", "35 to 55"),
    "55<=": ("٥٥ فأكثر", "55 and over"),
    "True": ("نعم", "yes"),
    "False": ("لا", "no"),
}


def say(key, **kw):
    """One label, in whichever language is showing."""
    pair = TEXT[key]
    text = pair[0] if st.session_state["lang"] == "ar" else pair[1]
    if kw:
        text = text.format(**kw)
    return text

def plain_date(text):
    """A date and time marked left to right.

    Without the marks, a browser reorders digits and dashes inside Arabic.
    """
    return "⁦" + str(text) + "⁩"


def word(name):
    """The plain name of one indicator."""
    pair = WORDS.get(name, (name, name))
    return pair[0] if st.session_state["lang"] == "ar" else pair[1]

# ---------------------------------------------------------------
# Reading

@st.cache_resource
def load_model():
    """The trained model, the six indicator names, the qualification list,
    the threshold and the cut day. They are saved together so the model is
    never fed the wrong columns in the wrong order."""
    return joblib.load(settings.project_file("final_model.pkl"))


def qualification(value):
    """One qualification, in the language on screen."""
    if pd.isna(value):
        return say("no_value")
    if st.session_state.get("lang", "ar") != "ar":
        return str(value)
    return QUALIFICATIONS.get(str(value), str(value))


def unit(name):
    """The unit one indicator is measured in, or an empty string."""
    pair = UNITS.get(name)
    if pair is None:
        return ""
    return pair[0] if st.session_state.get("lang", "ar") == "ar" else pair[1]


def value_shown(row, name):
    """One indicator's value, written the way a person reads it."""
    value = row.get(name)
    if name == "highest_education":
        return qualification(value)
    if pd.isna(value):
        return say("no_value")
    text = format(float(value), ".0f")
    return text + " " + unit(name) if unit(name) else text



def read_cohort(source):
    """Read a cohort file and build one row per student per course.

    The file holds one row per event. The system drops events after the cut
    day, builds the six indicators, and leaves an indicator empty when the
    file gives nothing to build it from.

    Returns four values: the table, the indicators it could not build, the
    key columns the file does not carry, and the columns holding text where
    a number belongs, with a count of such cells in each.
    """
    try:
        if isinstance(source, str):
            events = pd.read_csv(io.StringIO(source))
        else:
            events = pd.read_csv(source, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # A file exported from Excel on Windows is often not UTF-8, so the
        # usual Windows encoding is tried next.
        source.seek(0)
        events = pd.read_csv(source, encoding="cp1256")

    # The columns the file must carry. The rule lives in indicators.py.
    missing_keys = build.missing_keys(events)
    if missing_keys:
        return None, [], missing_keys, {}

    # Text where a number belongs. The cell is counted before the
    # derivation reads it, because the derivation sees only a gap.
    unreadable = {}
    for name in settings.NUMBER_COLUMNS:
        if name in events.columns and events[name].dtype == object:
            bad = int((events[name].notna()
                       & pd.to_numeric(events[name],
                                       errors="coerce").isna()).sum())
            if bad:
                unreadable[name] = bad

    table = build.from_events(events)
    table["rows_in_file"] = len(events)

    missing = [name for name in INDICATORS
               if name not in table.columns or table[name].isna().all()]
    return table, missing, [], unreadable


def tile(number, label, colour, under=""):
    """One counting tile. `under` is a second, smaller line below the
    label, used for a share."""
    css = "tile " + colour if colour else "tile"
    tail = '<div class="u">' + under + '</div>' if under else ""
    return ('<div class="' + css + '"><div class="n">' + str(number)
            + '</div><div class="t">' + label + '</div>' + tail + '</div>')

def register_rows(people, found):
    """One row per student, ready to be put on screen."""
    rows = []
    for who in people:
        mine = found[found["id_student"] == who]
        bad = mine[mine["flagged"]]
        rows.append({
            say("student"): str(who),
            say("col_failing"): len(bad),
            say("col_courses"): len(mine),
            # Keyed by the student number as text, because the database returns
            # text and the cohort file returns a number.
            say("state"): (say("sent")
                           if str(who) in st.session_state["sent"]
                           else say("waiting")),
        })

    table = pd.DataFrame(rows)

    def colour_state(value):
        if value == say("sent"):
            return "background-color:#eaf4ef;color:#1b8354;font-weight:600"
        return "color:#6b7280"

    return table.style.map(colour_state, subset=[say("state")])


def indicators_read(row):
    """How many of the indicators carried a value for this one record."""
    seen = 0
    for name in INDICATORS:
        if pd.notna(row[name]):
            seen = seen + 1
    return seen

# ---------------------------------------------------------------
# The scan

def scan(table, box, carried):
    """Run the scan. `carried` is how many of the six indicators were built.

    Every record gets one probability against one threshold.
    """
    found = table.copy()

    # A file with no indicator at all is not sent to the model: it would
    # return the same probability for every student.
    found["probability"] = np.nan
    if carried > 0:
        frame = build.for_the_model(
            found, education_levels=box["education_levels"])
        found["probability"] = box["model"].predict_proba(frame)[:, 1]

    found["flagged"] = settings.flags(found["probability"])

    # how much of the picture each record actually carried
    counts = []
    for i in range(len(found)):
        counts.append(indicators_read(found.iloc[i]))
    found["indicators_read"] = counts

    return found


def keep_scan(file_name, table, found, carried):
    """Store the file, the run and every flagged record.

    The reason is stored as written, so an old scan reads as it did then.
    """
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # The uploaded file is stored as it arrived, so it can be scanned again.
    contents = st.session_state.get("file_text", "")
    file_id = database.save_file(file_name, contents, len(found), when)

    # Every student in the file gets a row, flagged or not, because the
    # notice table points at this one.
    for who, part in table.groupby("id_student"):
        database.save_student(str(who), when)

    people = found.groupby("id_student")["flagged"].any()

    run_id = database.save_run(
        file_id=file_id, ran_on=when, day_of_term=settings.CUT_DAY,
        threshold=settings.ALERT_THRESHOLD, students=len(people),
        records=len(found), flagged=int(people.sum()),
        indicators_carried=carried)

    rows = []
    for i in range(len(found)):
        row = found.iloc[i]
        if not row["flagged"]:
            continue
        # The reason is written down now rather than worked out later, so a
        # stored scan does not change if the model is retrained.
        said = both_languages(model_reason_line, row, load_model())

        rows.append({
            "student_id": str(row["id_student"]),
            "course_code": str(row["code_module"]) + " "
                           + str(row["code_presentation"]),
            "reason": said["ar"],
            "reason_en": said["en"],
            "probability": float(row["probability"]),
            "indicators_read": int(row["indicators_read"]),
        })
    database.save_results(run_id, rows)

    return run_id


def both_languages(maker, row, model=None):
    """One reason, written in Arabic and in English, because a stored
    record may be read in either language."""
    was = st.session_state["lang"]
    out = {}
    for lang in ["ar", "en"]:
        st.session_state["lang"] = lang
        out[lang] = maker(row) if model is None else maker(row, model)
    st.session_state["lang"] = was
    return out


def stored_reason(part):
    """The stored reason, in the language on screen. Rows without an English
    column fall back to the one reason they have."""
    if st.session_state["lang"] == "ar" or "reason_en" not in part.columns:
        return part["reason"].values
    return [en if isinstance(en, str) and en != "" else ar
            for en, ar in zip(part["reason_en"].values,
                              part["reason"].values)]


def model_reasons(row, box, how_many=3):
    """Which indicators pushed this record towards an alert.

    Each push is reported as its share of the total push on the record.
    """
    one = build.for_the_model(row.to_frame().T,
                              education_levels=box["education_levels"])
    pushes = box["model"].predict(one, pred_contrib=True)[0][:len(INDICATORS)]

    total = float(np.sum(np.abs(pushes)))
    if total == 0:
        return []

    order = np.argsort(-pushes)       # strongest push towards risk first
    out = []
    for i in order[:how_many]:
        if pushes[i] <= 0:
            continue
        name = INDICATORS[i]
        value = row[name]

        # Each indicator is shown with its unit, and an empty cell is named.
        shown = value_shown(row, name)

        out.append({
            "name": word(name),
            "share": round(abs(pushes[i]) / total * 100, 1),
            "value": value,
            "shown": shown,
        })
    return out

def group_words(attribute, group):
    """A short name for one audited group, to fit under a bar.

    Looks in the short chart names, then the table values, then the label.
    """
    key = "grp_" + attribute + "_" + group
    if key in TEXT:
        return say(key)
    side = 0 if st.session_state.get("lang", "ar") == "ar" else 1
    if group in CELLS:
        return CELLS[group][side]
    return group


def model_reason_line(row, box):
    """The same reason as one line of text, used by the stored scan and by
    the notice so both carry the same wording."""
    parts = []
    for push in model_reasons(row, box):
        parts.append(push["name"] + " — " + push["shown"]
                     + " (" + str(push["share"]) + "%)")
    joiner = "، " if st.session_state["lang"] == "ar" else ", "
    return joiner.join(parts)

# ---------------------------------------------------------------
# The page

# The browser tab, set before the language is known, so it carries both.
st.set_page_config(page_title="الكشف المبكر عن التعثر · Academic Risk",
                   page_icon="🎓", layout="wide",
                   initial_sidebar_state="collapsed")

if "lang" not in st.session_state:
    st.session_state["lang"] = "ar"
# The tables are created before anything is read from them.
database.create_tables()

if "sent" not in st.session_state:
    # Read back from the database, because a notice sent earlier still stands.
    st.session_state["sent"] = database.notices_already_sent()
if "found" not in st.session_state:
    st.session_state["found"] = None
if "file_now" not in st.session_state:
    st.session_state["file_now"] = ""
if "file_text" not in st.session_state:
    # The uploaded file as it arrived, kept for storing and matching.
    st.session_state["file_text"] = ""
if "run_id" not in st.session_state:
    st.session_state["run_id"] = None
if "stored" not in st.session_state:
    st.session_state["stored"] = None
if "stored_when" not in st.session_state:
    st.session_state["stored_when"] = ""

if st.session_state["lang"] == "ar":
    # Arabic reads right to left. The tables are left alone, because turning
    # the grid round pushes the first column off the screen.
    st.markdown("""
        <style>
        .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
        .stApp li, .stApp .stMarkdown, .stApp .stCaption {
            direction: rtl; text-align: right;
        }
        /* RTL: the tab strip, buttons and radio rows start from the right. */
        .stApp .stTabs div:has(> [data-testid="stTab"]){
            flex-direction:row-reverse}
        .stApp .stTabs [data-testid="stTab"]{direction:rtl}
        /* RTL: a horizontal radio starts from the right. */
        .stApp [data-testid="stRadioGroup"][role="radiogroup"]:not(
            [data-baseweb="radio-group"]){flex-direction:row-reverse}
        .stApp [data-testid="stRadioOption"]{flex-direction:row-reverse}
        /* RTL: the radio is aligned with the surrounding content. */
        .stApp [data-testid="stElementContainer"]:has(
            > [data-testid="stRadio"]){align-self:flex-end}
        .stApp [data-testid="stRadio"] [data-testid="stWidgetLabel"] p{
            text-align:right}
        .stApp .stButton{display:flex;justify-content:flex-end}
        .stApp [data-testid="stFileUploaderDropzone"]{flex-direction:row-reverse}
        .stApp [data-testid="stFileUploaderDropzone"] *{direction:rtl}
        .stApp [data-testid="stFileUploaderFile"]{direction:rtl}
        /* The uploader's own English labels are replaced with Arabic. */
        .stApp [data-testid="stFileUploaderDropzone"]
            [data-testid="stMarkdownContainer"] p{font-size:0}
        .stApp [data-testid="stFileUploaderDropzone"]
            [data-testid="stMarkdownContainer"] p::after{
                content:"ارفع";font-size:14px}
        .stApp [data-testid="stFileUploaderDropzoneInstructions"] span{
            font-size:0}
        .stApp [data-testid="stFileUploaderDropzoneInstructions"]
            span::after{content:"حتى ٢٠٠ ميجابايت للملف · صيغة مفصولة "
                "بفواصل";font-size:13px}
        /* RTL: the table toolbar is moved clear of the heading. */
        .stApp [data-testid="stElementToolbar"]{inset-inline-end:auto;
            inset-inline-start:0}
        .stApp [data-testid="stDataFrame"],
        .stApp [data-testid="stDataFrame"] * {
            direction: ltr; text-align: left;
        }
        .stApp [data-testid="stMetricValue"] { direction: ltr; }
        </style>
    """, unsafe_allow_html=True)

st.markdown(STYLE, unsafe_allow_html=True)

@st.cache_data
def logo_tag():
    """The university logo, read from file and placed into the page."""
    try:
        with open(settings.project_file("university_logo.svg"), "rb") as handle:
            raw = handle.read()
        encoded = base64.b64encode(raw).decode()
        return '<img src="data:image/svg+xml;base64,' + encoded + '" alt="">'
    except FileNotFoundError:
        return ""

# The bar carries the logo and the language switch, and nothing else.
switch, mark = st.columns([1, 5])
with switch:
    if st.button(say("lang"), key="lang_switch"):
        # Which tab the advisor is on, read before the language changes. The
        # strip's key carries the language, so the old key is still live here.
        # The tab is kept as a position, which means the same tab either way.
        here = st.session_state.get(
            "nav_" + st.session_state["lang"]
            + "_" + str(st.session_state["tab_nonce"]))
        names = [say("tab_scan"), say("tab_register"), say("tab_evidence")]
        if here in names:
            st.session_state["which_tab"] = names.index(here)

        st.session_state["lang"] = ("en" if st.session_state["lang"] == "ar"
                                    else "ar")
        st.rerun()
with mark:
    # The logo, and nothing beside it.
    st.markdown('<div class="nav"><span class="logo">' + logo_tag()
                + '</span></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<div class="kick">' + ("الجامعة · لجنة الإشراف الأكاديمي"
                            if st.session_state["lang"] == "ar"
                            else "University · Academic Supervision Committee")
    + '</div><h1>' + say("title") + '</h1>'
    '<div class="lede">' + say("subtitle") + '</div></div>',
    unsafe_allow_html=True)

box = load_model()

# The scan screen can send the advisor straight to the register.
#
# A tab strip remembers the advisor's own choice, and that memory wins
# over anything the code sets. So to move the strip, the code builds a
# new one with a new key. The tab is held as a position, not a name,
# because the name changes with the language.
if "which_tab" not in st.session_state:
    st.session_state["which_tab"] = 0
if "tab_nonce" not in st.session_state:
    st.session_state["tab_nonce"] = 0


def go_to_tab(position):
    """Open a tab from code, and leave the advisor's own clicks alone."""
    st.session_state["which_tab"] = position
    st.session_state["tab_nonce"] = st.session_state["tab_nonce"] + 1


TAB_NAMES = [say("tab_scan"), say("tab_register"), say("tab_evidence")]

# The language is part of the key: switching it renames every tab, so
# the strip has to be built again.
TAB_KEY = ("nav_" + st.session_state["lang"]
           + "_" + str(st.session_state["tab_nonce"]))

# Where the advisor is, read BEFORE the strip is built. Read after, the
# strip would open on the previous tab and a click would appear to do
# nothing. When the code moves the strip the key is new and empty, so
# the position set by go_to_tab is the one used.
showing = st.session_state.get(TAB_KEY)
if showing in TAB_NAMES:
    st.session_state["which_tab"] = TAB_NAMES.index(showing)

# on_change="rerun" makes the strip report which tab was chosen.
# Without it a click is handled in the browser and never reaches here.
tab_scan, tab_register, tab_evidence = st.tabs(
    TAB_NAMES, default=TAB_NAMES[st.session_state["which_tab"]],
    key=TAB_KEY, on_change="rerun")

# ---------------------------------------------------------------
with tab_scan:
    # Where the cohort comes from. Each option says what it is underneath.
    source = st.radio(
        say("source"), [say("src_file"), say("src_db")],
        captions=[say("src_file_sub") + " · " + say("ready"),
                  say("src_db_sub") + " · " + say("not_ready")])
    on_file = source == say("src_file")

    if not on_file:
        st.info(say("db_closed"))

    else:
        # The key is fixed and the label is not part of it: a label that changes
        # with the language would be taken for a different widget.
        handed = st.file_uploader(say("upload"), type=["csv"],
                                  key="cohort_upload")

        # The uploader accepts a dropped file but says so in English only.
        st.caption(say("drag"))

        if handed is None:
            st.caption(say("no_file"))
        else:
            # A cohort file saved from Arabic Excel is often not UTF-8, so the two
            # likely Windows encodings are tried as well.
            raw = handed.getvalue()
            contents = None
            for encoding in ["utf-8", "utf-8-sig", "cp1256"]:
                try:
                    contents = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            readable = contents is not None
            if not readable:
                st.error(say("bad_encoding"))
                contents = ""

            # Kept so the file can be stored and recognised again later.
            st.session_state["file_text"] = contents

            # A different file clears the screen, so the name and the register agree.
            if st.session_state["file_now"] != handed.name + contents[:200]:
                st.session_state["file_now"] = handed.name + contents[:200]
                st.session_state["found"] = None
                st.session_state["sent"] = database.notices_already_sent()
                # The result opened from the database is cleared too.
                st.session_state["stored"] = None
                st.session_state["stored_when"] = ""

            # Decoding is not the same as parsing. A file that is not a cohort file
            # fails here, so the reading is guarded too.
            table = None
            missing_keys = []
            if readable and contents.strip() == "":
                # A file with nothing in it is named as empty rather than
                # leaving the screen with no file, no error and no button.
                st.error(say("empty_file"))
            elif readable:
                try:
                    table, missing, missing_keys, unreadable = read_cohort(
                        io.StringIO(contents))
                except Exception:
                    st.error(say("bad_file"))
                if table is not None and len(table) == 0:
                    st.error(say("empty_file"))
                    table = None

            # A file missing a key column is stopped here and the column is named,
            # because a screen that does nothing and says nothing cannot be acted on.
            if missing_keys:
                rows = ('<div class="gaphead">' + say("gap_no_keys")
                        + '</div><div class="gaplist">'
                        + "<br>".join("\u00b7 " + word(x)
                                      for x in missing_keys)
                        + '</div><div class="gaplist">'
                        + say("why_stopped") + '</div>')
                st.markdown('<div class="box stopped">' + rows + '</div>',
                            unsafe_allow_html=True)

            # A file that could not be read stops the scan, and nothing else.
            if table is not None:
                carried = len(INDICATORS) - len(missing)

                # Is this file already stored? Matched on contents, not on name.
                kept = st.session_state.get("file_text", "")
                already = database.find_file(kept)
                newer_than = database.other_versions(handed.name, kept)

                if already is not None:
                    st.info(say("file_known",
                                when=plain_date(str(already["uploaded_on"])),
                                rows=int(already["rows_in_file"])))
                elif len(newer_than) > 0:
                    st.warning(say("file_updated", n=len(newer_than)))

                # Said once, and only when the file is short of something. A file that
                # yields no indicator at all is not sent to the model, because it would
                # return the same probability for every student.
                stop = carried == 0

                if stop:
                    why = say("gap_none")
                    rows = ('<div class="gaphead">' + why + '</div>'
                            '<div class="gaplist">' + say("why_stopped")
                            + '</div>')
                    st.markdown('<div class="box stopped">' + rows
                                + '</div>', unsafe_allow_html=True)
                elif len(missing) > 0:
                    rows = ('<div class="gaphead">'
                            + say("gap_some", have=carried,
                                  total=len(INDICATORS))
                            + '</div><div class="gaplist">'
                            + "<br>".join("· " + word(x) for x in missing)
                            + '</div><div class="gaplist">'
                            + say("gap_some_note") + '</div>')
                    st.markdown('<div class="box warn">' + rows + '</div>',
                                unsafe_allow_html=True)

                # Text where a number belongs: the column was read but some cells were not.
                if len(unreadable) > 0:
                    rows = ('<div class="gaphead">' + say("gap_text")
                            + '</div><div class="gaplist">'
                            + "<br>".join("· " + word(x) + " — "
                                          + count_cells(unreadable[x])
                                          for x in unreadable)
                            + '</div><div class="gaplist">'
                            + say("gap_text_note") + '</div>')
                    st.markdown('<div class="box warn">' + rows + '</div>',
                                unsafe_allow_html=True)

                # A stored file can be scanned again, and the caption says what happens
                # to the first scan.
                if already is not None and not stop:
                    st.caption(say("run_again_note"))
                button_says = say("run_again") if already is not None \
                    else say("run")

                # The button stays on screen and goes dim rather than disappearing.
                if st.button(button_says, type="primary", disabled=stop):
                    with st.spinner(say("scanning")):
                        found = scan(table, box, carried)
                        st.session_state["found"] = found
                        # Kept for the facts line: which file, and when it was scanned.
                        st.session_state["file_now_name"] = handed.name
                        st.session_state["ran_on"] = \
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        st.session_state["run_id"] = keep_scan(
                            handed.name, table, found, carried)
                    st.rerun()

    found = st.session_state["found"]
    if found is not None:
        people = found.groupby("id_student")["flagged"].any()

        # The three stages of a scan, as the advisor sees them.
        steps = ""
        for number, key in enumerate(["step1", "step2_", "step3_"],
                                     start=1):
            steps = steps + ('<div class="sp2 done"><span class="dot">'
                             + str(number) + '</span>'
                             '<span class="tx">' + say(key) + '</span></div>')
        st.markdown('<div class="steps">' + steps + '</div>',
                    unsafe_allow_html=True)

        flagged_count = int(people.sum())
        share = (flagged_count / len(people) * 100) if len(people) else 0

        # Three tiles. The middle one is a button, the other two are drawn.
        # The order is reversed in Arabic so the count stays on the right.
        order = [0, 1, 2] if st.session_state["lang"] != "ar" else [2, 1, 0]
        boxes = st.columns(3)

        with boxes[order[0]]:
            st.markdown(tile(len(people), say("students"), "",
                             under="&nbsp;"), unsafe_allow_html=True)

        with boxes[order[1]]:
            face = ("**" + str(flagged_count) + "**" + say("flagged")
                    + "*" + format(share, ".1f") + "% "
                    + say("of_cohort") + "*")
            if st.button(face, key="to_register",
                         disabled=flagged_count == 0):
                go_to_tab(1)
                st.rerun()

        with boxes[order[2]]:
            st.markdown(tile(int((~people).sum()), say("clear"), "g",
                             under="&nbsp;"), unsafe_allow_html=True)

        # What this scan was: which file, cut where, at what threshold.
        facts = [
            (say("f_file"), st.session_state.get("file_now_name", "")
             + " · " + st.session_state.get("ran_on", "")),
            (say("f_records"), str(len(found))),
            (say("f_cut"), str(settings.CUT_DAY)),
            (say("f_threshold"), format(settings.ALERT_THRESHOLD, ".2f")),
        ]
        rows = ""
        for label, value in facts:
            rows = rows + ('<div class="push"><span>' + label
                           + '</span><span class="v">' + value
                           + '</span></div>')
        st.markdown('<div class="box">' + rows + '</div>',
                    unsafe_allow_html=True)

    # Everything scanned so far, so an earlier cohort can be opened again.
    st.divider()
    st.subheader(say("stored"))

    runs = database.get_runs()
    if len(runs) == 0:
        st.caption(say("no_stored"))
    else:
        shelf = pd.DataFrame({
            say("col_file"): runs["file_name"],
            say("col_when"): runs["ran_on"],
            say("col_students"): runs["students"],
            say("col_flagged"): runs["flagged"],
            say("col_carried"): runs["indicators_carried"],
        })
        chosen_run = st.dataframe(shelf, hide_index=True, width="stretch",
                                  on_select="rerun",
                                  selection_mode="single-row",
                                  key="runs_table")

        picked_rows = chosen_run["selection"]["rows"]
        if len(picked_rows) > 0:
            run = runs.iloc[picked_rows[0]]
            open_it, scan_again = st.columns(2)

            if open_it.button(say("open_run"), key="open_run"):
                st.session_state["stored"] = database.get_results(
                    int(run["run_id"]))
                st.session_state["stored_when"] = str(run["ran_on"])
                st.session_state["found"] = None
                st.rerun()

            if scan_again.button(say("rescan"), key="rescan"):
                contents = database.get_file_contents(int(run["file_id"]))
                st.session_state["file_text"] = contents
                table, missing, _, _ = read_cohort(io.StringIO(contents))
                stored_carried = len(INDICATORS) - len(missing)
                with st.spinner(say("scanning")):
                    found = scan(table, box, stored_carried)
                    st.session_state["found"] = found
                    st.session_state["file_now_name"] = str(
                        run["file_name"])
                    # A new scan carries the time it ran, not the time of the run it opened.
                    st.session_state["ran_on"] = \
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state["stored"] = None
                    st.session_state["run_id"] = keep_scan(
                        str(run["file_name"]), table, found, stored_carried)
                st.rerun()

# ---------------------------------------------------------------
with tab_register:
    found = st.session_state["found"]
    stored = st.session_state["stored"]

    if found is None and stored is not None:
        # A result read back from the database, exactly as it was written.
        st.info(say("from_store",
                    when=plain_date(st.session_state["stored_when"])))

        for heading in [say("flagged_block")]:
            part = stored
            if len(part) == 0:
                continue
            # Sorted by the student number as a number, not as text.
            part = part.copy()
            part["_n"] = pd.to_numeric(part["student_id"], errors="coerce")
            part = part.sort_values(["_n", "course_code"])

            # The heading counts students and course records separately.
            st.markdown('<div class="blockhead">' + heading
                        + ' <span class="sm">· '
                        + count_students(part["student_id"].nunique())
                        + " · " + count_records(len(part))
                        + '</span></div>',
                        unsafe_allow_html=True)

            # A probability is written to two places; a missing one shows empty.
            chance = []
            for v in part["probability"].values:
                chance.append("" if pd.isna(v) else format(float(v), ".2f"))

            shown = pd.DataFrame({
                say("student"): part["student_id"].values,
                say("col_course"): part["course_code"].values,
                say("col_reason"): stored_reason(part),
                say("col_prob"): chance,
                say("col_read"): part["indicators_read"].values,
            })
            st.dataframe(shown, hide_index=True, width="stretch")

    elif found is None:
        st.caption(say("no_file"))
    else:
        # One list: every student on it was flagged the same way.
        flagged_people = found[found["flagged"]]["id_student"].unique()

        # A search box, for a register of a hundred and fifty students.
        looking_for = st.text_input(say("search"), "",
                                    label_visibility="collapsed",
                                    placeholder=say("search"))

        picked = None
        shown = [w for w in flagged_people
                 if looking_for.strip() == ""
                 or looking_for.strip() in str(w)]

        if len(shown) > 0:
            st.markdown('<div class="blockhead">' + say("flagged_block")
                        + ' <span class="sm">· ' + count_students(len(shown))
                        + '</span></div>',
                        unsafe_allow_html=True)

            table = register_rows(shown, found)
            chosen = st.dataframe(
                table, hide_index=True, width="stretch",
                on_select="rerun", selection_mode="single-row",
                # The search text is part of the key, so a new search starts unselected.
                key="reg_" + looking_for.strip())

            # The returned row position belongs to the table as drawn, so it is
            # checked against the current list before being used.
            rows_picked = chosen["selection"]["rows"]
            if len(rows_picked) > 0 and rows_picked[0] < len(shown):
                picked = shown[rows_picked[0]]

        if picked is None and looking_for.strip() != "":
            if not any(looking_for.strip() in str(w)
                       for w in flagged_people):
                st.caption(say("nobody"))

        # A cohort with nobody flagged would otherwise leave this part blank.
        if len(flagged_people) == 0:
            st.info(say("all_clear"))
        else:
            st.caption(say("click_row"))

        # One student, opened.
        if picked is not None:
            who = picked
            mine = found[found["id_student"] == who]

            st.markdown('<div class="crumb">' + say("back") + " › <b>"
                        + say("student") + " " + str(who) + "</b></div>",
                        unsafe_allow_html=True)

            for i in range(len(mine)):
                row = mine.iloc[i]
                if not row["flagged"]:
                    continue

                # The threshold is given beside the probability.
                inside = ('<p class="why">' + say("prob") + ": <b>"
                          + format(float(row["probability"]), ".2f")
                          + "</b> " + say("against_threshold",
                                          t=format(settings.ALERT_THRESHOLD,
                                                   ".2f"))
                          + "</p>")
                pushes = model_reasons(row, box)
                # Each column is named, so a share and a value are not confused.
                if len(pushes) > 0:
                    inside = inside + (
                        '<div class="push hd"><span>'
                        + say("col_indicator") + '</span>'
                        '<span class="v">' + say("col_value") + '</span>'
                        '<span class="s">' + say("col_share") + '</span>'
                        '</div>')
                for push in pushes:
                    inside = inside + (
                        '<div class="push"><span>' + push["name"] + '</span>'
                        '<span class="v">' + push["shown"] + '</span>'
                        '<span class="s">' + str(push["share"]) + '%</span>'
                        '</div>')
                # And what the shares add up to.
                carried = round(sum(x["share"] for x in pushes), 1)
                inside = inside + ('<p class="note">'
                                   + say("push_note", n=len(pushes),
                                         total=carried) + '</p>')

                st.markdown(
                    '<div class="box">'
                    '<h3>' + str(row["code_module"]) + " "
                    + str(row["code_presentation"]) + '</h3>'
                    '<div class="cap">'
                    + count_indicators(int(row["indicators_read"]),
                                       len(INDICATORS))
                    + '</div>' + inside + '</div>',
                    unsafe_allow_html=True)

            if str(who) in st.session_state["sent"]:
                st.success(say("sent_at") + ": "
                           + plain_date(st.session_state["sent"][str(who)]))
            else:
                if st.button(say("send"), key="send_" + str(who), type="primary"):
                    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state["sent"][str(who)] = when
                    for i in range(len(mine)):
                        row = mine.iloc[i]
                        if not row["flagged"]:
                            continue
                        database.save_notice(
                            student_id=str(who),
                            course_code=str(row["code_module"]) + " "
                                        + str(row["code_presentation"]),
                            made_on=when,
                            day_of_term=settings.CUT_DAY,
                            reason=model_reason_line(row, box),
                            probability=float(row["probability"]),
                            indicators_read=int(row["indicators_read"]),
                            state="sent")
                    st.rerun()
            st.caption(say("stop_note"))

# ---------------------------------------------------------------
with tab_evidence:
    # The result files are written in English. Column names and cell values
    # are both translated for the screen.
    HEADINGS = {
        "Split": ("كيف قُسِّمت البيانات", "Split"),
        "Threshold": ("الحد الفاصل", "Threshold"),
        "Accuracy": ("الإصابة الكلية", "Accuracy"),
        "Precision": ("الدقة", "Precision"),
        "Recall": ("الاستدعاء", "Recall"),
        "F1": ("المتوسط التوافقي", "F1"),
        "ROC_AUC": ("المساحة تحت منحنى ROC", "ROC AUC"),
        "PR_AUC": ("المساحة تحت منحنى الدقة والاستدعاء", "PR AUC"),
        "Who decides": ("من يقرر", "Who decides"),
        "Rows_flagged": ("سجلات منذَر عنها", "Records flagged"),
        "What the model could see": ("ماذا يرى النموذج",
                                     "What the model could see"),
        "Columns": ("عدد الأعمدة", "Columns"),
        "Flagged": ("منذَر عنهم", "Flagged"),
        "Flagged_%": ("نسبة المنذَر عنهم", "Flagged %"),
        "Caught": ("ملقوطون", "Caught"),
        "Missed": ("مفقودون", "Missed"),
        "False_alarms": ("إنذارات كاذبة", "False alarms"),
        "Fold_spread": ("تذبذب الطيّات", "Fold spread"),
        "Note": ("ملاحظة", "Note"),
        "Attribute": ("الخاصية", "Attribute"),
        "Group": ("الفئة", "Group"),
        "Students": ("عدد الطلاب", "Students"),
        "Model": ("الخوارزمية", "Model"),
        "Share_%": ("نصيبه من القرار %", "Share of the decision %"),
        "Feature": ("المؤشر", "Indicator"),
        "Importance": ("الوزن الخام", "Raw importance"),
        "Missed_students": ("طلاب مفقودون", "Students missed"),
        "Indicators": ("عدد المؤشرات", "Indicators"),
        "Dataset": ("مجموعة البيانات", "Dataset"),
        "CV_recall": ("استدعاء التحقق المتقاطع", "CV recall"),
        "CV_Recall": ("استدعاء التحقق المتقاطع", "CV recall"),
        "AUC": ("المساحة تحت منحنى ROC", "ROC AUC"),
        "CV_spread": ("تذبذب الطيّات", "CV spread"),
        "Day": ("يوم الفصل", "Day of term"),
        "Rows_available": ("السجلات المتاحة", "Rows available"),
        "Sample_rows": ("السجلات المستخدمة", "Rows used"),
        "At_risk_share": ("نسبة المتعثرين", "At risk share"),
        # the columns of results_indicator_check.csv
        "candidate": ("المؤشر", "Indicator"),
        "gain_in_cross_validation": ("ما أضافه عند اختياره",
                                     "What it added when chosen"),
        "auc_without_it": ("المساحة تحت المنحنى بدونه", "ROC-AUC without it"),
        "it_adds_to_auc": ("ما يضيفه على التحقق", "What it adds on validation"),
        "at_risk_it_gains": ("متعثرون يكسبهم", "At-risk it gains"),
        "at_risk_it_loses": ("متعثرون يخسرهم", "At-risk it loses"),
        "net_at_risk": ("الصافي", "Net"),
        "rows_with_no_value": ("طلاب بلا قيمة فيه",
                               "Students with no value in it"),
        "decisions_moved_with_no_value": ("قرارات يحركها لمن لا قيمة لهم",
                                          "Decisions it moves for them"),
        "verdict": ("هل بقي", "Did it stay"),
        "indicator": ("المؤشر", "Indicator"),
        "share_of_gain": ("نصيبه من القرار", "Share of the decision"),
        "Test_rows": ("سجلات الاختبار", "Test rows"),
    }

    def dress(frame):
        """Put a result table into the language on screen."""
        out = frame.copy()
        # An empty cell is left empty. Streamlit prints a missing number as the
        # word None, so a column with a gap becomes text throughout; a column
        # holding both numbers and blanks cannot be rendered.
        for name in out.columns:
            if out[name].isna().any():
                out[name] = out[name].map(
                    lambda v: "" if pd.isna(v) else str(v)).astype("string")
        # The threshold is written to two places, as it is everywhere else.
        if "Threshold" in out.columns:
            out["Threshold"] = [format(float(v), ".2f")
                                for v in out["Threshold"]]
        side = 0 if st.session_state["lang"] == "ar" else 1
        out = out.replace({k: v[side] for k, v in CELLS.items()})
        out.columns = [HEADINGS.get(c, (c, c))[side] for c in out.columns]
        return out

    def show(file_name, caption, height=None):
        try:
            frame = pd.read_csv(settings.project_file(file_name))
        except FileNotFoundError:
            st.caption(caption + " — " + file_name)
            return None
        st.caption(caption)
        if height is None:
            st.dataframe(dress(frame), hide_index=True, width="stretch")
        else:
            st.dataframe(dress(frame), hide_index=True, width="stretch",
                         height=height)
        return frame

    # What these tables are, said before the first of them.
    st.info(say("ev_what"))

    # ---- 1. where the indicators came from
    st.markdown("#### " + say("ev_chosen"))
    st.caption(say("ev_chosen_note"))
    chosen = show("results_indicator_check.csv", say("ev_chosen_table"))

    if chosen is not None and "it_adds_to_auc" in chosen.columns:
        kept = chosen[chosen["verdict"] == "keeps its place"]
        if len(kept):
            st.caption(say("chart_chosen"))
            most = float(kept["gain_in_cross_validation"].max())
            bars = ('<div class="push hd"><span>' + say("col_indicator")
                    + '</span><span class="s">' + say("col_added")
                    + '</span></div>')
            for i in range(len(kept)):
                gain = float(kept["gain_in_cross_validation"].iloc[i])
                bars = bars + (
                    '<div class="wrow"><span class="nm">'
                    + word(str(kept["candidate"].iloc[i])) + '</span>'
                    '<span class="track"><span class="fill" style="width:'
                    + format(gain / most * 100, ".1f") + '%"></span></span>'
                    '<span class="s">' + format(gain, ".4f") + '</span></div>')
            st.markdown('<div class="box">' + bars + '</div>',
                        unsafe_allow_html=True)

    # ---- 2. and not one column about who the student is
    st.divider()
    st.markdown("#### " + say("ev_identity"))
    st.caption(say("ev_identity_note"))
    st.markdown("- " + "\n- ".join([
        say("id_gender"), say("id_region"), say("id_imd"),
        say("id_age"), say("id_disability")]))
    st.caption(say("ev_identity_one"))

    # ---- 3. what the model leans on
    st.divider()
    st.markdown("#### " + say("ev_weight"))
    st.caption(say("chart_weight"))
    try:
        weight = pd.read_csv(settings.project_file(
            "results_what_it_leans_on.csv"))
    except FileNotFoundError:
        weight = None

    if weight is None:
        st.caption(say("ev_no_file"))
    else:
        # The bar is drawn inside the table, because a chart alongside clips
        # the names.
        widest = float(weight["share_of_gain"].max())
        bars = ('<div class="push hd"><span>' + say("col_indicator")
                + '</span><span class="s">' + say("col_share")
                + '</span></div>')
        for i in range(len(weight)):
            share = float(weight["share_of_gain"].iloc[i])
            bars = bars + (
                '<div class="wrow"><span class="nm">'
                + word(str(weight["indicator"].iloc[i])) + '</span>'
                '<span class="track"><span class="fill" style="width:'
                + format(share / widest * 100, ".1f") + '%"></span></span>'
                '<span class="s">' + format(share, ".1f") + '%</span></div>')
        st.markdown('<div class="box">' + bars + '</div>',
                    unsafe_allow_html=True)

    # ---- 4. why this algorithm
    st.divider()
    st.markdown("#### " + say("ev_models_head"))
    st.caption(say("ev_models_note"))
    show("results_model_comparison.csv", say("ev_models"))

    # ---- 5. why this threshold
    st.divider()
    st.markdown("#### " + say("ev_thresh_head"))
    st.caption(say("ev_thresh_note"))
    threshold = show("results_threshold_table.csv", say("ev_thresh"))
    if threshold is not None and "Recall" in threshold.columns:
        st.caption(say("chart_threshold"))
        # The columns are renamed before the chart, which takes its legend
        # from them.
        line = threshold.set_index("Threshold")[["Recall", "Precision"]]
        line.columns = [say("m_recall"), say("m_precision")]
        line.index.name = say("col_threshold")
        st.line_chart(line, color=["#1b8354", "#b45309"], height=320)

    # ---- 6. does it treat everyone the same
    st.divider()
    st.markdown("#### " + say("ev_fair_head"))
    st.caption(say("ev_fair_note"))
    fair = show("results_fairness.csv", say("ev_fair"))
    if fair is not None and "Precision" in fair.columns:
        st.caption(say("ev_small"))
        st.caption(say("chart_fair"))
        # Only scored groups are drawn: a bar of zero would read as a measured
        # recall of zero.
        drawn = fair.copy()
        for name in ["Recall", "Precision"]:
            drawn[name] = pd.to_numeric(drawn[name], errors="coerce")
        drawn = drawn.dropna(subset=["Recall", "Precision"])
        # Each bar is named by attribute and group, and the two measures stand
        # side by side rather than stacked.
        drawn[say("col_group")] = [
            group_words(str(a), str(g))
            for a, g in zip(drawn["Attribute"], drawn["Group"])]
        # Laid on its side so each name has its own line, with the axis pinned
        # from 0 to 1. Both measures are proportions, and the plain bar chart
        # cannot pin an axis.
        tall = drawn.melt(id_vars=[say("col_group")],
                          value_vars=["Recall", "Precision"],
                          var_name="measure", value_name="value")
        tall = tall.rename(columns={say("col_group"): "group"})
        tall["measure"] = tall["measure"].map(
            {"Recall": say("m_recall"), "Precision": say("m_precision")})

        order = list(drawn[say("col_group")])
        both = [say("m_recall"), say("m_precision")]
        st.altair_chart(
            alt.Chart(tall).mark_bar().encode(
                x=alt.X("value:Q", title=None,
                        scale=alt.Scale(domain=[0, 1], nice=False)),
                y=alt.Y("group:N", title=None, sort=order),
                yOffset=alt.YOffset("measure:N", sort=both),
                color=alt.Color(
                    "measure:N", title=None,
                    scale=alt.Scale(domain=both,
                                    range=["#1b8354", "#b45309"]))),
            height=360)

    # ---- 7. the result
    st.divider()
    st.markdown("#### " + say("ev_test_head"))
    st.caption(say("ev_test_note"))
    show("results_final.csv", say("ev_test"))
