"""
The interface of the Smart Academic Risk Detection System.

Three screens, and the order between them is the order of the work:

  1. Scan      - read a cohort file and find the students in trouble
  2. Register  - one row per student, and what the trouble is per course
  3. Evidence  - how well the system does, from the result files

WHAT THIS SCREEN DOES AND WHERE IT STOPS
The system reads a cohort's records, builds the six indicators, asks the
model about every student, and tells the advisor who is at risk and what the
model read to say so. Then it stops. It does not advise the advisor what to
do, it does not record what the advisor decided, and it does not follow the
case afterwards. Those belong to the advisor and the student.

ONE JUDGEMENT PER STUDENT
Every record is judged once, by the model, on the same six indicators. There
is no second opinion to reconcile and no order of precedence to explain: one
probability, one threshold, one answer, and the indicators that carried it
shown beside it.

WHAT THE FILE CARRIES AND WHAT THE SYSTEM DOES WITH IT
The uploaded file is one row per event - a student registered, a piece of
work handed in, a day's clicks on a page - exactly as a university system
exports its records. The advisor aggregates nothing. The system cuts
everything dated after the cut day, builds the six indicators itself, and
scores.

NOTHING IS INVENTED
Every reason shown beside a student is computed from that student's own
record. Where the file carries nothing to build an indicator from, the cell
is left empty and the model is told it is empty rather than given a zero. A
zero means "measured as none"; empty means "not known", and the model was
trained to tell those apart.
"""

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

# Plain words for each indicator, in both languages. The model's column names
# are for the code; these are what a person reads.
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
}

# The qualification values as the records carry them, and what each one is
# called in Arabic. The data is published in English, and an English word
# sitting inside an Arabic sentence is the one thing this screen must not
# do, so each value is given its Arabic name here. A value not listed is
# shown as it stands rather than guessed at.
QUALIFICATIONS = {
    "No Formal quals":             "بدون مؤهل رسمي",
    "Lower Than A Level":          "أقل من الثانوية",
    "A Level or Equivalent":       "الثانوية أو ما يعادلها",
    "HE Qualification":            "مؤهل جامعي",
    "Post Graduate Qualification": "مؤهل دراسات عليا",
}

# The unit each indicator is measured in, so a number is never printed on
# screen without saying what it is a number of.
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
    # --- the seven sections of the performance page, in the order they
    #     are read: where the indicators came from, what the model reads,
    #     why this algorithm, why this threshold, is it even-handed, and
    #     last of all the result.
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
                         "أياً منها. لم نستبعدها — التجربة لم تجد فيها "
                         "ما يضيف.",
                         "These columns were offered to the same "
                         "experiment and it chose none of them. We did "
                         "not exclude them; the experiment found nothing "
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

    "ev_thresh":    ("كل حد فاصل جربناه — مجموعة التحقق، لأن عليها اختِير الحد",
                     "Every threshold we tried — validation set, because that "
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
                     "خليّة فارغة تعني أننا لم نقِسها، لا أنها صفر.",
                     "A group of fewer than fifty students is counted but "
                     "not scored: an empty cell means we did not measure "
                     "it, not that it is zero."),
    "lang":         ("English", "العربية"),
    "src_file_sub": ("ملف الدفعة بصيغة مفصولة بفواصل",
                     "A cohort file in comma separated form"),
    "src_db_sub":   ("قراءة مباشرة من سجلات الجامعة",
                     "Read straight from the university's records"),
    "ready":        ("متاح", "available"),
    "not_ready":    ("غير متاح", "not available"),
    # The three stages of a scan, as the advisor sees them. What the
    # system does inside each one is its own business.
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

    # Column headings. Each one names what is in its own column and
    # nothing else: a heading describes the column, not the control that
    # produced it.
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
                        "اخترنا ٠٫٢٠ لأن الطالب المفقود يخسر فصلاً، "
                        "والإنذار الكاذب يكلّف مكالمة، وما يُفعل بالإنذار "
                        "قرار المرشد لا قرار النظام.",
                        "How recall and precision trade off at each "
                        "threshold. We chose 0.20 because a student we "
                        "miss loses a term, and a false alarm costs a "
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
    """A count of students, written the way each language writes it.

    Arabic does not put a plural after every number the way English does:
    one is طالب واحد, two is طالبان, three to ten take the plural طلاب,
    and eleven upwards take the singular again in the accusative, طالباً.
    Printing "94 طلاب" is the kind of mistake that makes a careful reader
    stop trusting the rest of the screen, so the four cases are written
    out here once and used everywhere a count of students is shown.
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

# The look of the system. These are the same colours, font and shapes as the
# simulation the committee was shown, so the working system and the
# simulation are recognisably the one thing. The green is the university's.
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
/* The font is set here and left to be inherited. It is deliberately NOT
   forced onto every element: Streamlit draws its small icons with a font
   where a word like "upload" is a picture rather than letters, so an
   element that sets its own font has to be allowed to keep it. Forcing
   ours everywhere printed the word "upload" beside the button. */
html, body, .stApp { font-family: var(--font); }
button, input, select, textarea { font-family: var(--font); }
.stApp [data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded'!important}
.stApp { background: var(--band); color: var(--ink); }
.block-container { padding-top: 1.2rem; max-width: 1280px; }
#MainMenu, footer, header { visibility: hidden; }
/* the band across the top, the same one the simulation opens with */
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
/* The three boxes stand side by side and must be the same height. Each
   has a slot for a second line under the label; the two that have
   nothing to put there leave it empty rather than leaving it out, so
   all three are built the same and line up without being measured. */
.tile .n{font-size:36px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.tile .t{font-size:14.5px;color:var(--ink-2);margin-top:8px}
.tile.r .n{color:var(--red)} .tile.o .n{color:var(--orange)} .tile.g .n{color:var(--green)}
/* The flagged tile is a real button, because a count of students is
   only useful if the students are one click away - and a drawn box
   cannot be clicked. It is a Streamlit button wearing the tile's
   clothes: same border, same radius, same padding, the number in the
   same size and the same orange.
   It is reached by the key it was given. Wrapping it in markup of our
   own does not work: Streamlit draws every element in its own box, so
   an opening tag written before the button and a closing tag after it
   end up as two empty boxes with the button between them, outside
   both. The key is the one handle the page really has on it.
   A button holds one line of text, so the three lines are made with
   bold and italic and sized by what they are rather than by a class.
   Its height is pinned to the drawn tiles beside it, because a button
   and a div do not add up their padding the same way and three boxes
   in a row that differ by a few pixels look like a mistake.
   NOTE: no blank line may appear anywhere inside this stylesheet, not
   even inside a comment. A blank line ends the block of HTML and the
   rest of the sheet is printed onto the page as words. */
.st-key-to_register{width:100%}
.st-key-to_register .stButton{display:block;width:100%}
.st-key-to_register .stButton>button{width:100%!important;
  min-height:136px;background:var(--paper);
  border:1px solid var(--line);border-radius:12px;padding:22px 16px;
  text-align:center;box-shadow:none;transition:border-color .15s,
  box-shadow .15s}
.st-key-to_register .stButton>button:hover{border-color:var(--orange);
  box-shadow:0 2px 10px rgba(180,83,9,.12)}
.st-key-to_register .stButton>button:focus{box-shadow:none;
  border-color:var(--orange)}
.st-key-to_register .stButton>button p{margin:0;line-height:1.35;
  color:var(--ink-2);font-size:14.5px;font-weight:400}
.st-key-to_register .stButton>button p strong{display:block;
  font-size:36px;font-weight:700;line-height:1;color:var(--orange);
  font-variant-numeric:tabular-nums;margin-bottom:8px}
.st-key-to_register .stButton>button p em{display:block;
  font-style:normal;font-size:12px;color:var(--muted);margin-top:4px}
.tile .u{font-size:12px;color:var(--muted);margin-top:4px}
.box.warn{border-inline-start:3px solid var(--orange);
  background:var(--orange-l)}
.box.stopped{border-inline-start:3px solid var(--red);
  background:var(--red-l)}
.box .gaphead{font-weight:700;margin-bottom:6px}
.box .gaplist{color:var(--ink-2);font-size:14px;line-height:1.95}
/* the register, laid out the way the simulation lays it out */
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
/* one indicator and the share of the decision it carries, drawn here so
   the bar is the same green as the rest of the system and the name has a
   whole line to itself instead of being cut off on a chart's edge */
.wrow{display:flex;align-items:center;gap:14px;padding:9px 0;
      border-bottom:1px solid var(--band);font-size:14px}
.wrow:last-child{border-bottom:0}
.wrow .nm{flex:0 0 220px}
.wrow .track{flex:1 1 auto;height:9px;border-radius:5px;background:var(--band)}
.wrow .fill{display:block;height:9px;border-radius:5px;background:var(--green)}
.wrow .s{flex:0 0 72px;text-align:end;font-weight:700;color:var(--green);
         font-variant-numeric:tabular-nums}
/* the bar across the very top, with the university's mark on it */
/* The mark sits on the right in both languages. It is the university's
   own artwork, not a piece of writing, so it does not change sides with
   the text. Forcing the bar's direction is what keeps it there when the
   rest of the screen turns around. */
.nav{background:var(--paper);border:1px solid var(--line);border-radius:12px;
     padding:12px 18px;display:flex;align-items:center;margin-bottom:14px;
     direction:rtl}
.nav .logo img{height:34px;width:auto;display:block}
/* the five steps of the scan */
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
/* the scan log */
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
/* Streamlit paints two small things in its own red: the bar under the
   chosen tab, and the dot of the chosen radio button. Both are set to the
   university's green here rather than in a settings file, so the whole
   look of the system travels with app.py and nothing else has to be
   copied beside it. */
.stApp .react-aria-SelectionIndicator{background:var(--green)!important}
.stApp [data-testid="stRadioOption"]:has(input:checked) > div > div:first-child{
  background:var(--green)!important}
</style>
"""

# The values inside the result tables, in both languages. It sits
# here, beside the other word tables, because the chart under a
# table reads it too - and a name must not say one thing in a table
# and another in the chart beneath it.
CELLS = {
    "random (final test)": ("الشريحة المحجوزة — فُتحت مرة واحدة",
                            "the held-out slice, opened once"),
    # the verdict column of the selection table
    "keeps its place": ("بقي", "stayed"),
    "not confirmed": ("لم يُثبت نفسه", "did not hold up"),
    # The three columns the fairness audit reads, and their values.
    # The published data writes them as gender / F / True; on screen
    # they are written the way a committee would say them.
    "gender": ("الجنس", "gender"),
    "region": ("المنطقة", "region"),
    "imd_band": ("المستوى الاجتماعي", "deprivation band"),
    # The regions as the data publishes them. An English place name
    # inside an Arabic table is still an English word on the page.
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
    """A date and time that reads the same in both languages.

    A browser laying out Arabic reorders a run of digits and dashes by its
    own rules, so "2026-11-08 16:17" came out as "16:17 08-11-2026" in the
    middle of an Arabic sentence: the time first and the day and year
    swapped. Marking the stamp left to right stops that, and the same
    stamp then reads the same wherever it appears.
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
    """
    The trained model and what it needs in order to read a cohort.

    The six indicator names, the qualification list from the training
    data, the threshold and the cut day are saved alongside the model
    itself. They travel together on purpose: a model separated from the
    list of what it was trained on can be fed the wrong columns in the
    wrong order without anything going visibly wrong.
    """
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

# The three columns that identify a record: which student, which course,
# which term. Everything else the system can do without.
KEY_COLUMNS = ["id_student", "code_module", "code_presentation"]


def read_cohort(source):
    """Read a cohort file and build one row per student per course.

    The file is one row per event - a student registered, a piece of work
    handed in, a day's clicks on a page - the way a university system
    exports its records. The advisor aggregates nothing and uploads no
    table of indicators: the system cuts everything dated after the cut
    day, builds the six indicators itself, and scores.

    An indicator the file gives nothing to build from is left empty, not
    set to zero. The model was trained to read an empty cell as "not
    known" and to decide which way to send it, so telling it the truth is
    better than filling the gap with a number nobody measured.

    Returns the built table, the indicators it could not build, the
    identity columns the file does not carry, and nothing unreadable -
    because a value that cannot be read becomes an empty cell during the
    building and is counted there.
    """
    try:
        if isinstance(source, str):
            events = pd.read_csv(io.StringIO(source))
        else:
            events = pd.read_csv(source, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # A file exported from Excel on Windows is often not in UTF-8.
        # Reading it blindly throws an error about bytes, which tells the
        # advisor nothing, so the usual Windows encoding is tried before
        # giving up.
        source.seek(0)
        events = pd.read_csv(source, encoding="cp1256")

    # The columns the file must carry before anything can be read. The
    # rule is in indicators.py, beside the reading it guards.
    missing_keys = build.missing_keys(events)
    if missing_keys:
        return None, [], missing_keys, {}

    table = build.from_events(events)
    table["rows_in_file"] = len(events)

    missing = [name for name in INDICATORS
               if name not in table.columns or table[name].isna().all()]
    return table, missing, [], {}


def tile(number, label, colour, under=""):
    """One counting tile.

    `under` is a second, quieter line beneath the label - a share, where
    a bare count does not say much on its own. 151 means little; 151 out
    of 217 means something.
    """
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
            # Keyed by the student number as text. The database returns
            # it as text and the cohort file reads it as a number, and a
            # notice sent yesterday has to be recognised either way.
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
    """The whole scan, in the order the system is built on.

    carried is how many of the six indicators the file let us build.

    Every record goes to the model and gets one probability against one
    threshold. There is no record that skips it and no record decided
    some other way, which is why the result needs no column saying how it
    was reached.
    """
    found = table.copy()

    # A file that let us build no indicator at all is not put to the
    # model. Asked about a row of empty cells the model answers from the
    # pattern it learned and nothing else, so it returns the same number
    # for every student in the file - which would flag or clear a whole
    # cohort with nothing behind it. The scan says so instead.
    found["probability"] = np.nan
    if carried > 0:
        frame = build.for_the_model(
            found, education_levels=box["education_levels"])
        found["probability"] = box["model"].predict_proba(frame)[:, 1]

    found["flagged"] = found["probability"] >= settings.ALERT_THRESHOLD

    # how much of the picture each record actually carried
    counts = []
    for i in range(len(found)):
        counts.append(indicators_read(found.iloc[i]))
    found["indicators_read"] = counts

    return found


def keep_scan(file_name, table, found, carried):
    """Store the file, the run and everything the run flagged.

    The file is stored so it can be scanned again later, trimmed to the
    columns the system reads. The results go in with the reason already
    written out, so an old scan reads exactly as it read on the day rather
    than being worked out again from a model that may have changed since.
    """
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # The uploaded file is stored as it arrived, so the same scan can be
    # run again later. It carries events and nothing identifying a person
    # beyond the student number the university already holds.
    contents = st.session_state.get("file_text", "")
    file_id = database.save_file(file_name, contents, len(found), when)

    # Every student the file carries, flagged or not, gets a row. The
    # notice table points at this one, so a notice with no student behind
    # it would be a notice about nobody. A student on two courses has one
    # row here and two records in the scan, which is the whole point of
    # keeping the two apart.
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
        # The reason is the handful of indicators that pushed this record
        # towards an alert, and it is written down here rather than worked
        # out again later. A stored scan has to read as it read on the
        # day: the model may be retrained, and an old notice must not
        # quietly change its mind afterwards.
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
    """One reason, written in Arabic and in English.

    A stored record outlives the language it was written in, and the
    language on screen at the moment of a scan says nothing about the
    language it will be read in. Both are written down, and the screen
    picks the one it needs.
    """
    was = st.session_state["lang"]
    out = {}
    for lang in ["ar", "en"]:
        st.session_state["lang"] = lang
        out[lang] = maker(row) if model is None else maker(row, model)
    st.session_state["lang"] = was
    return out


def stored_reason(part):
    """The stored reason, in the language on screen.

    Rows written before the English column existed fall back to the one
    reason they have, rather than showing an empty cell.
    """
    if st.session_state["lang"] == "ar" or "reason_en" not in part.columns:
        return part["reason"].values
    return [en if isinstance(en, str) and en != "" else ar
            for en, ar in zip(part["reason_en"].values,
                              part["reason"].values)]


def model_reasons(row, box, how_many=3):
    """Which indicators pushed this record towards an alert, and by how much.

    LightGBM gives the push of each indicator in log odds, which is not a
    number to show anybody. So each push is reported as its share of the
    total push on this record: a plain statement of which indicators carried
    the decision, without pretending the units are marks or percentages.
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

        # Each indicator is shown in the unit it is measured in, so a
        # number never appears on screen without saying what it counts.
        # An empty cell is named rather than printed as a dash: for this
        # system an absence is a reading, and the strongest one it has.
        shown = value_shown(row, name)

        out.append({
            "name": word(name),
            "share": round(abs(pushes[i]) / total * 100, 1),
            "value": value,
            "shown": shown,
        })
    return out

def group_words(attribute, group):
    """A name for one audited group, short enough to read under a bar.

    The audit's own labels are F, M, True, False, two age bands, the
    thirteen regions and the ten deprivation bands. Put on a chart as
    they are, half of them say nothing and the other half are English
    words on an Arabic page, so each one is given a short plain name.
    They are kept short on purpose: a long label is turned on its edge
    by the chart and then cut off.

    Three places are looked in, in order: the short chart name, then the
    same table of values the tables themselves are translated with, then
    the label as it stands. The middle one is what keeps a name from
    reading one way in a table and another way in the chart beneath it.
    """
    key = "grp_" + attribute + "_" + group
    if key in TEXT:
        return say(key)
    side = 0 if st.session_state.get("lang", "ar") == "ar" else 1
    if group in CELLS:
        return CELLS[group][side]
    return group


def model_reason_line(row, box):
    """The same reason, written as one line of text.

    A stored scan and a sent notice have to carry the same justification,
    because they are two records of one decision. Writing the line in one
    place is what keeps them from drifting apart.
    """
    parts = []
    for push in model_reasons(row, box):
        parts.append(push["name"] + " — " + push["shown"]
                     + " (" + str(push["share"]) + "%)")
    joiner = "، " if st.session_state["lang"] == "ar" else ", "
    return joiner.join(parts)

# ---------------------------------------------------------------
# The page

# The browser tab. Streamlit fixes this before the language is known, so
# it carries both names rather than claiming the page is only English.
st.set_page_config(page_title="الكشف المبكر عن التعثر · Academic Risk",
                   page_icon="🎓", layout="wide",
                   initial_sidebar_state="collapsed")

if "lang" not in st.session_state:
    st.session_state["lang"] = "ar"
# The tables are made before anything is read from them, because the very
# first thing this screen does is ask which notices have already been sent.
database.create_tables()

if "sent" not in st.session_state:
    # Read back from the database rather than started empty. A notice that
    # was sent yesterday is a thing that happened, and closing the browser
    # does not unhappen it.
    st.session_state["sent"] = database.notices_already_sent()
if "found" not in st.session_state:
    st.session_state["found"] = None
if "file_now" not in st.session_state:
    st.session_state["file_now"] = ""
if "file_text" not in st.session_state:
    # The uploaded file as it arrived, kept so it can be stored exactly as
    # it was read and recognised again on a later upload.
    st.session_state["file_text"] = ""
if "run_id" not in st.session_state:
    st.session_state["run_id"] = None
if "stored" not in st.session_state:
    st.session_state["stored"] = None
if "stored_when" not in st.session_state:
    st.session_state["stored_when"] = ""

if st.session_state["lang"] == "ar":
    # Arabic reads right to left, so the writing is turned round. The tables
    # are left alone on purpose: a table of student numbers and percentages
    # is read left to right in both languages, and turning the grid round
    # pushes its first column off the edge of the screen.
    st.markdown("""
        <style>
        .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
        .stApp li, .stApp .stMarkdown, .stApp .stCaption {
            direction: rtl; text-align: right;
        }
        /* The tab strip, the buttons and the radio rows are laid out by
           Streamlit from the left. In Arabic they belong on the right,
           with the first tab under the start of the heading above it. */
        .stApp .stTabs div:has(> [data-testid="stTab"]){
            flex-direction:row-reverse}
        .stApp .stTabs [data-testid="stTab"]{direction:rtl}
        /* A radio laid out across the screen starts from the left; in
           Arabic it starts from the right. One laid out down the screen
           is already in order and only its dots move to the right side. */
        .stApp [data-testid="stRadioGroup"][role="radiogroup"]:not(
            [data-baseweb="radio-group"]){flex-direction:row-reverse}
        .stApp [data-testid="stRadioOption"]{flex-direction:row-reverse}
        /* The radio is only as wide as its choices, so left to itself it
           sits against the left edge while everything around it starts
           from the right. Pushed over to begin where the rest begins. */
        .stApp [data-testid="stElementContainer"]:has(
            > [data-testid="stRadio"]){align-self:flex-end}
        .stApp [data-testid="stRadio"] [data-testid="stWidgetLabel"] p{
            text-align:right}
        .stApp .stButton{display:flex;justify-content:flex-end}
        .stApp [data-testid="stFileUploaderDropzone"]{flex-direction:row-reverse}
        .stApp [data-testid="stFileUploaderDropzone"] *{direction:rtl}
        .stApp [data-testid="stFileUploaderFile"]{direction:rtl}
        /* Streamlit writes the uploader's own button and its size note in
           English and gives no way to change them. An English word sitting
           inside an Arabic line is the one thing this screen must not do,
           so the English is hidden and the Arabic put in its place. */
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
        /* Streamlit floats a little toolbar over the top corner of every
           table. In Arabic that corner is where the heading ends, so the
           toolbar was sitting on top of the words. Pushed clear of them. */
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
    """The university's mark, read from the file and put straight into the
    page. It is kept as a file rather than pasted into the code so the
    university's own artwork is not edited by us."""
    import base64
    try:
        raw = open(settings.project_file("university_logo.svg"), "rb").read()
        encoded = base64.b64encode(raw).decode()
        return '<img src="data:image/svg+xml;base64,' + encoded + '" alt="">'
    except FileNotFoundError:
        return ""

# The bar carries the university's mark and the language switch, and
# nothing else. There is no sign in and there are no user accounts here,
# so an account name beside the mark would claim something untrue.
#
# The language switch sits here, across from the mark: in reach without a
# panel of its own, and on the same side in both languages.
switch, mark = st.columns([1, 5])
with switch:
    if st.button(say("lang"), key="lang_switch"):
        # Which tab the advisor is on, read before the language changes.
        #
        # Clicking a tab does not run the program: the strip handles it
        # in the browser, and the only record of the choice is the
        # strip's own state, under the key it was built with. That key
        # carries the language in it, so a moment from now it will be a
        # different key with nothing in it. The tab is read here, while
        # the old key is still the live one, and kept as a position -
        # which means the same tab in either language.
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
    # The university's mark, and nothing beside it. The name of the
    # committee is already on the banner underneath, and writing it twice
    # on one screen said it no better the second time.
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

# The scan screen can send the advisor straight to the register: a count
# of flagged students is only useful if the students themselves are one
# click away.
#
# WHY THE KEY CHANGES
#   A tab strip remembers which tab the advisor chose, and that memory
#   wins over anything the code asks for afterwards - which is right,
#   because a screen that moved under someone's hand would be worse than
#   one that did not move at all. Writing the wanted tab into the strip's
#   own state does nothing: it stays where the advisor left it.
#
#   So when the code needs to move, it builds a NEW strip, with a new
#   key and the wanted tab as its opening one. The old strip's memory
#   belongs to the old key and does not apply. Between those moments the
#   advisor's own clicks are untouched, because the key only changes when
#   the code asks for a move.
#   The wanted tab is remembered by its POSITION, never by its name. A
#   name is the label on screen, and the label changes with the
#   language: a tab remembered as "سجل الطلاب" is not among the tabs a
#   moment after the advisor switches to English, and the page stops
#   with an error instead of turning around. A position means the same
#   thing in both languages.
if "which_tab" not in st.session_state:
    st.session_state["which_tab"] = 0
if "tab_nonce" not in st.session_state:
    st.session_state["tab_nonce"] = 0


def go_to_tab(position):
    """Open a tab from code, and leave the advisor's own clicks alone."""
    st.session_state["which_tab"] = position
    st.session_state["tab_nonce"] = st.session_state["tab_nonce"] + 1


TAB_NAMES = [say("tab_scan"), say("tab_register"), say("tab_evidence")]

# The language is part of the key as well as the counter. Switching
# language renames every tab, and a strip whose remembered tab no longer
# exists under its new name has to be built again.
TAB_KEY = ("nav_" + st.session_state["lang"]
           + "_" + str(st.session_state["tab_nonce"]))

# Where the advisor actually is, read BEFORE the strip is built.
#
# The order matters and it is the whole of it. A click on a tab sets the
# strip's own state and asks for a rerun; on that rerun the strip is
# built again, and the tab it opens on is whatever is handed to default.
# Read afterwards, the position would still be the one the advisor had
# left a moment ago, the strip would open there, and the click would
# appear to have done nothing - it took a second click to show. Read
# first, default is already the tab that was just clicked.
#
# When the code moves the strip itself the key is new, so there is
# nothing stored under it, nothing is read, and the position go_to_tab
# set is the one that stands. The language switch is the same case: the
# language is part of the key, so the new key is empty and the position
# it saved under the old key, above, is the one used.
showing = st.session_state.get(TAB_KEY)
if showing in TAB_NAMES:
    st.session_state["which_tab"] = TAB_NAMES.index(showing)

# on_change="rerun" is what makes the strip tell the program which tab
# was chosen. Left alone, a tab click is handled entirely in the browser
# and never reaches here, so the program has no way of knowing where the
# advisor is - and the next time the strip is rebuilt it would put them
# back wherever the code last left them.
tab_scan, tab_register, tab_evidence = st.tabs(
    TAB_NAMES, default=TAB_NAMES[st.session_state["which_tab"]],
    key=TAB_KEY, on_change="rerun")

# ---------------------------------------------------------------
with tab_scan:
    # Where the cohort comes from. One control, and what each source is
    # and whether it is open to us is written under its own option rather
    # than in a second place that would have to be kept in step.
    source = st.radio(
        say("source"), [say("src_file"), say("src_db")],
        captions=[say("src_file_sub") + " · " + say("ready"),
                  say("src_db_sub") + " · " + say("not_ready")])
    on_file = source == say("src_file")

    if not on_file:
        st.info(say("db_closed"))

    else:
        # The key is fixed and the label is not part of it. Without one,
        # changing language changed the uploader's label, Streamlit took
        # it for a different widget, and the file the advisor had just
        # scanned disappeared along with the result on screen.
        handed = st.file_uploader(say("upload"), type=["csv"],
                                  key="cohort_upload")

        # Streamlit's uploader takes a dropped file as well as a chosen one,
        # but says so only in English and only on the button. This line says
        # it in the language the rest of the screen is in.
        st.caption(say("drag"))

        if handed is None:
            st.caption(say("no_file"))
        else:
            # A cohort file saved out of Arabic Excel is often not written
            # in UTF-8, and reading it blindly throws an error about bytes
            # that tells the advisor nothing. The two encodings a Windows
            # machine is likely to produce are tried as well, and a file in
            # none of them is refused politely.
            raw = handed.getvalue()
            contents = None
            for encoding in ["utf-8", "utf-8-sig", "cp1256"]:
                try:
                    contents = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if contents is None:
                st.error(say("bad_encoding"))
                contents = ""

            # Kept so the file can be stored exactly as it arrived and so
            # the same text is used to recognise it again later.
            st.session_state["file_text"] = contents

            # A different file than last time clears the screen straight
            # away. Otherwise the name at the top would say one file while
            # the register underneath still showed another one's students.
            if st.session_state["file_now"] != handed.name + contents[:200]:
                st.session_state["file_now"] = handed.name + contents[:200]
                st.session_state["found"] = None
                st.session_state["sent"] = database.notices_already_sent()
                # The result opened from the database has to go as well.
                # Without this the register kept showing the students of an
                # older cohort while the scan screen named the new file, so
                # the two screens described two different cohorts at once.
                st.session_state["stored"] = None
                st.session_state["stored_when"] = ""

            # Decoding into letters is not the same as the letters making a
            # table. A file in the wrong format, or one that is not a cohort
            # file at all, fails here rather than at the decode, so the
            # reading is guarded too and the same plain message is shown.
            table = None
            missing_keys = []
            if contents != "":
                try:
                    table, missing, missing_keys, unreadable = read_cohort(
                        io.StringIO(contents))
                except Exception:
                    st.error(say("bad_file"))
                if table is not None and len(table) == 0:
                    st.error(say("empty_file"))
                    table = None

            # A file missing one of the columns the scan is built on is
            # stopped here and named here, before anything else. There is
            # no student to put in the register without a student number,
            # and no record to read without the column that says what each
            # row is. The advisor is told which column is absent, because a
            # screen that does nothing and says nothing cannot be acted on.
            if missing_keys:
                rows = ('<div class="gaphead">' + say("gap_no_keys")
                        + '</div><div class="gaplist">'
                        + "<br>".join("\u00b7 " + word(x)
                                      for x in missing_keys)
                        + '</div><div class="gaplist">'
                        + say("why_stopped") + '</div>')
                st.markdown('<div class="box stopped">' + rows + '</div>',
                            unsafe_allow_html=True)

            # A file that could not be read stops the scan, and nothing
            # more. The rest of the screen, including the shelf of files
            # scanned before, stays where it is: losing the whole page
            # because one upload was the wrong format helps nobody.
            if table is not None:
                carried = len(INDICATORS) - len(missing)

                # Is this file already here? The answer comes from its
                # contents, not its name, and from the same text that
                # gets stored - so an uploaded file matches what was
                # saved from it rather than never matching at all.
                kept = st.session_state.get("file_text", "")
                already = database.find_file(kept)
                newer_than = database.other_versions(handed.name, kept)

                if already is not None:
                    st.info(say("file_known",
                                when=plain_date(str(already["uploaded_on"])),
                                rows=int(already["rows_in_file"])))
                elif len(newer_than) > 0:
                    st.warning(say("file_updated", n=len(newer_than)))

                # Whether the file is short of anything, said once and
                # only when it is. A file that supplies all six says
                # nothing here: there is nothing to warn about, and a
                # count on a healthy file reads as though the system
                # discovers indicators in whatever it is handed, which it
                # does not. It knows six, and looks for those six.
                #
                # A file that was read but yields no indicator at all is
                # not put to the model. Handed a row of empty cells the
                # model answers from the pattern it learned and nothing
                # else, returning the same number for every student in the
                # file, which would flag or clear a whole cohort for no
                # stated reason. A file missing a column the scan needs was
                # stopped and named above, before this point.
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

                # Text where a number belongs is a separate complaint: the
                # column is there and was read, but some of its cells
                # could not be made into numbers.
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

                # A file the system already holds can be scanned again,
                # but the button says so and the line underneath says what
                # happens to the first scan. An advisor pressing a button
                # marked "scan" on a file already scanned has no way of
                # knowing whether the earlier one is about to be lost.
                if already is not None and not stop:
                    st.caption(say("run_again_note"))
                button_says = say("run_again") if already is not None \
                    else say("run")

                # The button stays where it is and goes dim. Taking it off
                # the screen would read as the page having broken.
                if st.button(button_says, type="primary", disabled=stop):
                    with st.spinner(say("scanning")):
                        found = scan(table, box, carried)
                        st.session_state["found"] = found
                        # Kept for the facts line: which file, scanned
                        # when. Read from here rather than worked out
                        # again, so the screen says what actually ran.
                        st.session_state["file_now_name"] = handed.name
                        st.session_state["ran_on"] = \
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        st.session_state["run_id"] = keep_scan(
                            handed.name, table, found, carried)
                    st.rerun()

    found = st.session_state["found"]
    if found is not None:
        people = found.groupby("id_student")["flagged"].any()

        # The three stages of a scan. What the system does inside each
        # one is its own business, and naming those steps on screen tells
        # an advisor nothing they can act on.
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

        # Three tiles. The middle one is a button and the other two are
        # drawn, because it is the only one worth opening: a student with
        # nothing against them has no record to read, and a total leads
        # to the same place the flagged count does.
        #
        # They sit in columns rather than in one drawn row, because a
        # Streamlit button cannot live inside markup we wrote ourselves.
        # The order is reversed in Arabic so the count of students stays
        # on the right, where the reading starts.
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

        # What this scan was, in four plain lines. Not a log: an advisor
        # reading this screen wants to know which file, cut where, at what
        # threshold - not the order the work was done in.
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

    # Everything scanned so far, so any earlier cohort can be opened again
    # or put through the system a second time.
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
                    # This is a new scan and it is stored as one, so the
                    # screen carries the moment it ran. Reading the date off
                    # the run it was opened from would put the older scan
                    # time on a result produced now.
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
        # A result opened from the database. It is read back exactly as it
        # was written, so there is no model here and nothing is worked out
        # again: these are the sentences the system gave on the day.
        st.info(say("from_store",
                    when=plain_date(st.session_state["stored_when"])))

        for heading in [say("flagged_block")]:
            part = stored
            if len(part) == 0:
                continue
            # Sorted by the student number as a number. Sorting it as
            # text put 1411627 before 141648 and left the committee
            # looking at an order nobody could explain.
            part = part.copy()
            part["_n"] = pd.to_numeric(part["student_id"], errors="coerce")
            part = part.sort_values(["_n", "course_code"])

            # The heading counts students and course records separately,
            # because this table has one row per record while the heading
            # above it counts students.
            st.markdown('<div class="blockhead">' + heading
                        + ' <span class="sm">· '
                        + count_students(part["student_id"].nunique())
                        + " · " + count_records(len(part))
                        + '</span></div>',
                        unsafe_allow_html=True)

            # A probability is written to two places so a column of them
            # can be read down at a glance, and a row stored before the
            # column existed shows an empty cell rather than the word
            # None.
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
        # One list. Every student on it was flagged the same way, by the
        # model, so there is nothing to sort them into.
        flagged_people = found[found["flagged"]]["id_student"].unique()

        # A search box. With a hundred and fifty students on the register,
        # an advisor looking for one of them needs it.
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
                # The search text is part of the key on purpose. Changing
                # the search builds a fresh table with nothing selected,
                # instead of carrying the old row's position over to a
                # different list of students.
                key="reg_" + looking_for.strip())

            # Streamlit hands back the position of the chosen row, and that
            # position belongs to the table as it was drawn. Typing in the
            # search box redraws a shorter table while the old position is
            # still remembered, so the position is checked against the list
            # it is about to be used on. Without this check a search could
            # throw an error onto the screen, or quietly open a different
            # student than the one whose row was clicked.
            rows_picked = chosen["selection"]["rows"]
            if len(rows_picked) > 0 and rows_picked[0] < len(shown):
                picked = shown[rows_picked[0]]

        if picked is None and looking_for.strip() != "":
            if not any(looking_for.strip() in str(w)
                       for w in flagged_people):
                st.caption(say("nobody"))

        # A cohort where nobody is flagged would otherwise leave this part
        # of the screen blank, which reads as a system that did not run
        # rather than one that ran and found nothing.
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

                # The threshold is given beside the probability. A number
                # on its own invites the question "compared to what", and
                # the answer is the only thing that makes the number mean
                # anything.
                inside = ('<p class="why">' + say("prob") + ": <b>"
                          + format(float(row["probability"]), ".2f")
                          + "</b> " + say("against_threshold",
                                          t=format(settings.ALERT_THRESHOLD,
                                                   ".2f"))
                          + "</p>")
                pushes = model_reasons(row, box)
                # Three numbers in a row mean nothing without saying what
                # each column is. A share of 23.9 per cent beside a value
                # of 63.5 per cent needs a word on the card to tell the
                # two apart.
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
                # And what they add up to, so nobody has to wonder what
                # happened to the rest of the decision.
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
    # The result files are written in English by the scripts that produce
    # them, because that is the language of the report. On screen they are
    # read by whoever is in the room, so the column names and the words
    # inside the cells are both turned into the language being shown.
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
        # An empty cell is left empty. Streamlit prints a missing number
        # as the word None, which reads as a value rather than as the
        # absence of one, and the absence is the thing being said. Only
        # the columns that actually have a gap are rewritten, so the rest
        # of the table stays numbers and stays lined up on the right.
        for name in out.columns:
            if out[name].isna().any():
                out[name] = ["" if pd.isna(v) else v for v in out[name]]
        # The threshold is written to two places on the student's card
        # and in the scan log, so it is written the same way here. One
        # number should not have two spellings inside one system.
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

    # What these tables are, said before the first of them. Without this
    # line a reader who has just scanned a cohort takes the numbers for
    # the result of that scan, which is the one thing they are not.
    st.info(say("ev_what"))

    # ---- 1. where the indicators came from
    #
    # This is first on the page on purpose. Every other number here rests
    # on the six indicators, so the first thing a reader is owed is where
    # those six came from - and the answer is a table, not a sentence.
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
    #
    # The strongest single claim the project makes, and the one a
    # committee is most likely to test. It is stated here as a result of
    # the experiment above, never as a position we took.
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
        # The bar is drawn inside the table rather than beside it. The
        # names are phrases, not labels, and a chart alongside clips them
        # after a few letters; in the row, the name has a whole column.
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
        # A chart takes its legend from the column names, so the columns
        # are renamed before it is drawn. Two English words in the corner
        # of an Arabic screen are still two English words.
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
        # Only the groups that were scored are drawn. A group left
        # unmeasured has no bar, because a bar of zero would say we
        # measured a recall of zero, which is not what the empty cell
        # in the table above means.
        drawn = fair.copy()
        for name in ["Recall", "Precision"]:
            drawn[name] = pd.to_numeric(drawn[name], errors="coerce")
        drawn = drawn.dropna(subset=["Recall", "Precision"])
        # Each bar is named by its attribute as well as its group, because
        # "True" and "F" on their own say nothing, and the two measures
        # stand side by side rather than stacked: a recall and a precision
        # added together make a number that means nothing.
        drawn[say("col_group")] = [
            group_words(str(a), str(g))
            for a, g in zip(drawn["Attribute"], drawn["Group"])]
        # Laid on its side, so each group's name has a line of its own
        # instead of being stood on its edge under a bar.
        #
        # The axis is pinned from nought to one, and it is drawn here
        # rather than by the plain bar chart because that one has no way
        # to pin it. Left to choose, it fits the axis to the bars and to
        # the width it is given, and the same figures came out once on an
        # axis of nought to one and once on an axis running from below
        # nought to past one and a half. Both measures are proportions.
        # Neither can be less than nothing or more than everything, and
        # an axis that says otherwise invites the reader to compare two
        # drawings of the same table and find them different.
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
    #
    # Last, and last on purpose. Read first it is a number with nothing
    # behind it; read here it is the outcome of everything above.
    st.divider()
    st.markdown("#### " + say("ev_test_head"))
    st.caption(say("ev_test_note"))
    show("results_final.csv", say("ev_test"))
