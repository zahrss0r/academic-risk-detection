"""
The interface of the Smart Academic Risk Detection System.

Three screens, and the order between them is the order of the work:

  1. Scan      - read a cohort file and find the students in trouble
  2. Register  - one row per student, and what the trouble is per course
  3. Evidence  - how well the system does, from the result files

WHAT THIS SCREEN DOES AND WHERE IT STOPS
The system reads a cohort, applies the university's regulations, asks the
model about the students no regulation caught, and tells the advisor. Then it
stops. It does not advise the advisor what to do, it does not record what the
advisor decided, and it does not follow the case afterwards. Those belong to
the advisor and the student.

THE ORDER IS STRICT
Every row meets the regulations first. A row a regulation catches is never
put to the model at all: no probability is computed for it and none is shown.
A regulation is arithmetic and arithmetic is not appealed to a probability.

NOTHING IS INVENTED
Every reason shown beside a student is computed from that student's own row.
Where the file does not carry a column, the rule that needs it says so and
stays silent, and the model is told the value is missing rather than given a
zero. A zero means "measured as none"; missing means "not known".
"""

import datetime
import io

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import settings
import database
import university_rules

INDICATORS = settings.MODEL_INDICATORS

# Plain words for each indicator, in both languages. The model's column names
# are for the code; these are what a person reads.
RULE_WORDS = {
    "work not handed in":    ("العمل غير المسلَّم", "work not handed in"),
    "absence":               ("الغياب", "absence"),
    "marks still reachable": ("الدرجة القصوى الممكنة",
                              "marks still reachable"),
    "the withdrawal window": ("نافذة الانسحاب", "the withdrawal window"),
}

WORDS = {
    "early_score":          ("الدرجة المكتسبة حتى الآن", "marks earned so far"),
    "early_score_missing":  ("لا توجد درجة مرصودة", "no mark recorded yet"),
    "submission_rate":      ("نسبة ما سُلِّم من المطلوب", "share of due work handed in"),
    "late_rate":            ("نسبة ما سُلِّم متأخراً", "share handed in late"),
    "late_rate_missing":    ("لم يُسلّم شيئاً", "nothing handed in at all"),
    "total_clicks":         ("مرات فتح المقرر", "times the course was opened"),
    "activity_days_rate":   ("نسبة أيام النشاط", "share of days active"),
    "longest_gap_days":     ("أطول انقطاع بالأيام", "longest gap in days"),
    "num_of_prev_attempts": ("محاولات سابقة في المقرر", "previous attempts"),
    "studied_credits":      ("الساعات المسجلة", "credits registered"),
    "late_registration":    ("تسجيل متأخر", "registered late"),
    "id_student":           ("رقم الطالب", "student number"),
    "code_module":          ("رمز المقرر", "course code"),
    "code_presentation":    ("الفصل الدراسي", "term"),
}

TEXT = {
    "title":        ("نظام الكشف المبكر عن التعثر الأكاديمي",
                     "Smart Academic Risk Detection System"),
    "subtitle":     ("يقرأ الدفعة، يطبّق اللائحة، ثم ينذر المرشد",
                     "Reads the cohort, applies the regulations, notifies the advisor"),
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
    "cols_read":    ("المؤشرات التي يحملها الملف",
                     "Indicators this file carries"),
    "gap_keys":     ("أعمدة الهوية غير الموجودة في الملف",
                     "Identity columns this file does not carry"),
    "gap_rules":    ("لوائح لا تستطيع العمل على هذا الملف، وعمود كل منها",
                     "Regulations this file cannot let speak, and the "
                     "column each one reads"),
    "caught_nobody": ("لم تمسك أحداً", "caught nobody"),
    "cannot_scan":  ("لا يمكن فحص هذا الملف",
                     "This file cannot be scanned"),
    "why_stopped":  ("لا يُنذَر عن طالب لا يُعرف رقمه، ولا يُبنى إنذار على "
                     "ملف لا يحمل مؤشراً واحداً ولا لائحة واحدة. أضف ما "
                     "ينقص أدناه وارفع الملف مرة أخرى.",
                     "The system does not warn about a student whose number "
                     "it does not know, and does not build an alert on a "
                     "file carrying neither a single indicator nor a single "
                     "regulation. Add what is listed below and upload the "
                     "file again."),
    "gap_text":     ("خانات غير قابلة للقراءة كرقم",
                     "Cells that could not be read as a number"),
    "gap_text_note": ("قُرئت كقيمة مجهولة، لا كصفر — كما يُقرأ أي نقص آخر.",
                      "They were read as not known, not as a zero, the "
                      "same as any other gap."),
    "gap_cols":     ("مؤشرات لا يحملها الملف",
                     "Indicators this file does not carry"),
    "cols_missing": ("أُبلغ النموذج أنها مجهولة ولم تُعطَ صفراً.",
                     "The model was told these are unknown, not given a "
                     "zero."),
    "students":     ("طلاب", "students"),
    "rows":         ("سجلات مقرر", "course records"),
    "flagged":      ("منذَر عنهم", "flagged"),
    "clear":        ("لا شيء عليهم", "clear"),
    "by_rule":      ("أمسكتهم اللائحة", "caught by a regulation"),
    "by_model":     ("أمسكهم النموذج", "caught by the model"),
    "rule_block":   ("اللائحة أمسكتهم — سبب حسابي يمكن التحقق منه",
                     "Caught by a regulation — arithmetic the advisor can check"),
    "model_block":  ("النموذج أمسكهم — لم تنطبق عليهم لائحة",
                     "Caught by the model — no regulation applied to them"),
    "student":      ("الطالب", "Student"),
    "courses":      ("المقررات", "courses"),
    "failing_in":   ("متعثر في", "in trouble in"),
    "pick":         ("اختر طالباً لعرض سجله", "Pick a student to open the record"),
    "why":          ("لماذا ظهر هذا الطالب", "Why this student appears"),
    "read_n":       ("قرأ النظام {n} من {total} مؤشرات لهذا السجل",
                     "The system read {n} of {total} indicators for this record"),
    "prob":         ("احتمال التعثر", "Probability"),
    "no_prob":      ("لا يُحسب احتمال لطالب أمسكته اللائحة",
                     "No probability is computed for a student a regulation caught"),
    "state":        ("حالة الإنذار", "Notice"),
    "waiting":      ("في الانتظار", "waiting"),
    "sent":         ("أُرسل للمرشد", "sent to the advisor"),
    "send":         ("أرسل للمرشد الأكاديمي", "Notify the academic advisor"),
    "sent_at":      ("أُرسل في", "Sent"),
    "stop_note":    ("ينتهي دور النظام هنا. ما يفعله المرشد بعد ذلك ليس من "
                     "شأن النظام ولا يُسجَّل فيه.",
                     "The system's part ends here. What the advisor does next "
                     "is not the system's business and is not recorded in it."),
    "rules_state":  ("حالة اللوائح على هذا الملف",
                     "What the regulations can reach in this file"),
    "rule_name":    ("اللائحة", "Regulation"),
    "rule_needs":   ("تحتاج", "needs"),
    "rule_says":    ("على هذا الملف", "on this file"),
    "speaks":       ("تتكلم", "speaks"),
    "silent":       ("صامتة — العمود غير موجود",
                     "silent — the column is not here"),
    "silent_ok":    ("صحيحة ولا تنطبق على أحد عند هذا اليوم",
                     "correct, and catches nobody on this day"),
    "ev_what":      ("هذه الأرقام تصف النموذج نفسه، لا الملف الذي تفحصه. "
                     "قيست مرة واحدة على ٢٨٢٨ سجلاً لم يرها النموذج أثناء "
                     "التدريب، ونُعرف نتيجتها. ولا تتغير بتغيّر الدفعة، "
                     "لأن الفصل الجاري لم تُعرف نتيجته بعد.",
                     "These numbers describe the model itself, not the file "
                     "you are scanning. They were measured once on 2,828 "
                     "records the model never saw in training, whose "
                     "outcome is known. They do not change when the cohort "
                     "changes, because a term still running has no outcome "
                     "yet."),
    "ev_test":      ("الاختبار النهائي — فُتح مرة واحدة",
                     "The final test — opened once"),
    "ev_valid":     ("مجموعة التحقق — عليها اختِيرت كل القرارات",
                     "The validation set — where every decision was made"),
    "ev_rule":      ("اللائحة أمام النموذج، على الاختبار النهائي",
                     "The regulation against the model, on the final test"),
    "ev_groups":    ("ماذا يرى النموذج — مجموعة التحقق",
                     "What the model can see — validation set"),
    "ev_thresh":    ("كل حد فاصل جربناه — مجموعة التحقق، لأن عليها اختِير الحد",
                     "Every threshold we tried — validation set, because that "
                     "is where the threshold was chosen"),
    "ev_models":    ("الخوارزميات الأربع عند حد ٠.٥٠ — مجموعة التحقق. "
                     "الأرقام هنا لا تقارن بالاختبار النهائي أعلاه.",
                     "The four algorithms at a threshold of 0.50 — validation "
                     "set. These numbers are not comparable with the final "
                     "test above."),
    "ev_fair":      ("فحص العدالة — على مجموعة التحقق",
                     "Fairness audit — on the validation set"),
    "ev_fair_note": ("هذه الأعمدة لا تُعطى للنموذج. تُستخدم لمراجعته فقط. "
                     "ومجموع كل خاصية ٣٧٨٢ لأن الفحص على مجموعة التحقق "
                     "لا الاختبار.",
                     "These columns are not given to the model. They are used "
                     "only to audit it. Each attribute totals 3,782 because "
                     "this audit runs on the validation set, not the test "
                     "set."),
    "ev_weight":    ("وزن كل مؤشر في القرار", "What each indicator carries"),
    "ev_timing":    ("اليوم الخمسون أمام اليوم المئة — مجموعة التحقق",
                     "Day fifty against day one hundred — validation set"),
    "ev_timing_note": ("نفس التجربة أُجريت عند منتصف الفصل وعند اليوم "
                       "الخمسين. الفحص عند اليوم الخمسين أضعف في كل رقم، "
                       "وهذا سبب اختيار اليوم المئة موعداً للفحص.",
                       "The same experiment was run at mid semester and at "
                       "day fifty. Every number is weaker at day fifty, and "
                       "that is why day one hundred is when the scan runs."),
    "ev_data":      ("نفس الطريقة على ثلاث مجموعات بيانات",
                     "The same method on three datasets"),
    "ev_data_note": ("الطريقة نفسها، والمؤشرات هي ما تحمله كل مجموعة. "
                     "النظام مبني على المجموعة الأولى؛ والاثنتان الأخريان "
                     "تقولان أين تصلح الطريقة وأين لا تصلح.",
                     "The same method, with whatever indicators each dataset "
                     "carries. The system is built on the first one; the "
                     "other two say where the method holds and where it "
                     "does not."),
    "ev_small":     ("الفئة التي عدد طلابها أقل من خمسين تُعدّ ولا تُقاس: "
                     "خليّة فارغة تعني أننا لم نقِسها، لا أنها صفر.",
                     "A group of fewer than fifty students is counted but "
                     "not scored: an empty cell means we did not measure "
                     "it, not that it is zero."),
    "lang":         ("English", "العربية"),
    "committee":    ("لجنة الإشراف الأكاديمي", "Academic Supervision Committee"),
    "role":         ("حساب اللجنة", "Committee account"),
    "src_file_sub": ("ملف الدفعة بصيغة مفصولة بفواصل",
                     "A cohort file in comma separated form"),
    "src_db_sub":   ("قراءة مباشرة من سجلات الجامعة",
                     "Read straight from the university's records"),
    "ready":        ("متاح", "available"),
    "not_ready":    ("غير متاح", "not available"),
    "step1":        ("قراءة الملف", "Read the file"),
    "step2":        ("تطبيق اللوائح", "Apply the regulations"),
    "step3":        ("سؤال النموذج", "Ask the model"),
    "step4":        ("تجميع الطلاب", "Gather by student"),
    "step5":        ("فتح السجل", "Open the register"),
    "search":       ("ابحث برقم الطالب", "Search by student number"),
    "chip_all":     ("الكل", "All"),
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
    "all_clear":    ("لا أحد على هذا الملف يستدعي إنذاراً: لا اللوائح "
                     "أمسكت أحداً، ولا النموذج أنذر عن أحد.",
                     "Nobody on this file needs an alert: no regulation "
                     "caught anyone, and the model flagged nobody."),
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
    "pick_run":     ("اختر عملية فحص", "Pick a scan"),
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

    # The four regulations, and what each one needs to be able to speak.
    # They are keyed by the English name the rules file gives them, so a
    # rule renamed there is caught here rather than silently mistranslated.
    "rule_work_not_handed_in":  ("عمل مستحق لم يُسلَّم", "work not handed in"),
    "rule_absence":             ("الغياب", "absence"),
    "rule_marks_still_reachable": ("أعلى درجة ما زالت ممكنة",
                                   "marks still reachable"),
    "rule_the_withdrawal_window": ("نافذة الاعتذار والانسحاب",
                                   "the withdrawal window"),
    "needs_submission_rate":    ("نسبة ما سُلِّم من المطلوب",
                                 "how much of the due work was handed in"),
    "needs_a_count_of_unexcused_absences": ("عدد الغيابات غير المعذورة",
                                            "a count of unexcused absences"),
    "needs_the_weight_of_each_assessment": ("وزن كل تقييم في المقرر",
                                            "the weight of each assessment"),
    "needs_the_date_of_the_final_exam":    ("تاريخ الاختبار النهائي",
                                            "the date of the final exam"),

    # Column headings. Each one names what is in its own column and
    # nothing else. An earlier version headed the file name with "a file
    # I upload" and the date with "pick a scan", which were the labels of
    # the controls that produced them, not descriptions of the columns.
    "col_file":     ("الملف", "File"),
    "col_when":     ("وقت الفحص", "Scanned at"),
    "col_students": ("عدد الطلاب", "Students"),
    "col_flagged":  ("منذَر عنهم", "Flagged"),
    "col_by_rule":  ("باللائحة", "By a regulation"),
    "col_by_model": ("بالنموذج", "By the model"),
    "col_carried":  ("مؤشرات الملف", "Indicators in the file"),
    "col_caught":   ("من أمسكه", "Caught by"),
    "col_failing":  ("مقررات متعثرة", "Courses in trouble"),
    "col_courses":  ("مقررات مسجَّلة", "Courses taken"),
    "col_course":   ("المقرر", "Course"),
    "col_reason":   ("السبب", "Reason"),
    "col_prob":     ("احتمال التعثر", "Probability"),
    "col_read":     ("مؤشرات مقروءة", "Indicators read"),
    "col_indicator": ("المؤشر", "Indicator"),
    "col_value":    ("قيمته على هذا السجل", "its value on this record"),
    "col_share":    ("نصيبه من القرار", "share of the decision"),
    "col_raw":      ("الوزن الخام", "raw importance"),
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
    "yes":          ("نعم", "yes"),
    "no":           ("لا", "no"),
    "against_threshold": ("مقابل حد الإنذار {t}", "against the alert threshold of {t}"),
    "push_note":    ("أقوى {n} مؤشرات دفعت نحو الإنذار، وتحمل {total}٪ من "
                     "القرار. الباقي موزَّع على بقية المؤشرات.",
                     "The {n} indicators that pushed hardest towards an "
                     "alert, carrying {total}% of the decision between "
                     "them. The rest is spread across the others."),

    # Captions for the three charts on the performance screen.
    "chart_threshold": ("كيف يتبادل الاستدعاء والدقة عند كل حد فاصل. "
                        "اخترنا ٠.٣٠ لأن الطالب المفقود يخسر فصلاً، "
                        "والإنذار الكاذب يكلّف مكالمة.",
                        "How recall and precision trade off at each "
                        "threshold. We chose 0.30 because a student we "
                        "miss loses a term, and a false alarm costs a "
                        "phone call."),
    "chart_fair":      ("الاستدعاء والدقة لكل فئة. الفجوة في الدقة بين "
                        "الإناث والذكور معروضة كما خرجت، ولم تُخفَ.",
                        "Recall and precision for each group. The "
                        "precision gap between female and male students "
                        "is shown as it came out, not hidden."),
    "chart_weight":    ("نصيب كل مؤشر من القرار. السلوك على المنصة يحمل "
                        "أكثر مما تحمله الدرجة المبكرة.",
                        "What each indicator carries of the decision. "
                        "Behaviour on the platform carries more than the "
                        "early mark does."),
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
      padding:22px 16px;text-align:center;min-width:0}
.tile .n{font-size:36px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.tile .t{font-size:14.5px;color:var(--ink-2);margin-top:8px}
.tile.r .n{color:var(--red)} .tile.o .n{color:var(--orange)} .tile.g .n{color:var(--green)}
/* the register, laid out the way the simulation lays it out */
.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;
      overflow:hidden;margin-bottom:20px}
.card .scroll{overflow-x:auto}
.card table{width:100%;border-collapse:collapse;font-size:15px}
.card thead th{text-align:start;font-weight:500;font-size:13px;color:var(--muted);
               padding:12px 18px;border-bottom:1px solid var(--line);white-space:nowrap}
.card tbody td{padding:14px 18px;border-bottom:1px solid var(--band)}
.card tbody tr:last-child td{border-bottom:0}
.card tr.rulerow td:first-child{box-shadow:inset 3px 0 0 var(--red)}
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
.log{background:#0d1b3e;color:#c9d6f5;border-radius:10px;padding:15px 17px;
     font-size:13px;line-height:2;font-variant-numeric:tabular-nums;
     margin-bottom:18px}
.log b{color:var(--green-br);font-weight:400}
.log .d{color:#7f91bf}
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

def block(heading, items, note=""):
    """One titled list, for a message that has several things to say.

    Three findings run together in a paragraph read as one long sentence
    and get skimmed. Under their own headings, each is a list an advisor
    can work through and tick off. A heading says what the list is in a
    few words; anything longer goes in the note underneath it, where it
    does not push the heading onto a second line.
    """
    out = "**" + heading + "**"
    if note:
        out = out + "\n\n" + note
    out = out + "\n"
    for item in items:
        out = out + "\n- " + item
    return out


def joiner():
    """The list separator of the language on screen."""
    return "، " if st.session_state["lang"] == "ar" else ", "


def rule_name(name):
    """The plain name of one regulation."""
    pair = RULE_WORDS.get(name, (name, name))
    return pair[0] if st.session_state["lang"] == "ar" else pair[1]


def word(name):
    """The plain name of one indicator."""
    pair = WORDS.get(name, (name, name))
    return pair[0] if st.session_state["lang"] == "ar" else pair[1]

# ---------------------------------------------------------------
# Reading

@st.cache_resource
def load_model():
    return joblib.load(settings.project_file("final_model.pkl"))

# The three columns that identify a record: which student, which course,
# which term. Everything else the system can do without.
KEY_COLUMNS = ["id_student", "code_module", "code_presentation"]


def read_cohort(source):
    """Read a cohort file and line its columns up with what the model needs.

    A file may not carry every indicator. Any it does not carry is added as
    an empty column, not a zero. The model was trained to read an empty cell
    as "not known" and to decide which way to send it, so telling it the
    truth is better than filling the gap with a number nobody measured.
    """
    table = pd.read_csv(source)

    # The three columns that say whose record this is. An indicator the
    # file does not carry becomes an empty column and the scan goes on
    # without it. One of these is different: without a student number
    # there is nobody to put in the register and nobody to notify, so
    # the scan has no subject at all. They are reported, not filled in.
    missing_keys = [n for n in KEY_COLUMNS if n not in table.columns]

    missing = []
    for name in INDICATORS:
        if name not in table.columns:
            table[name] = np.nan
            missing.append(name)

    # Every indicator must be a number. A file with the word "absent" in
    # a column of click counts used to reach the model and bring down the
    # whole page with a traceback. Text in a numbered column is read the
    # same way an empty cell is read: not known. That is the system's own
    # rule for a value it does not have, and it is applied here rather
    # than crashing or quietly treating the text as a zero. How many
    # cells this touched is counted, because an advisor whose file has
    # text in it should be told, not left to wonder.
    unreadable = {}
    for name in INDICATORS:
        before = table[name].notna().sum()
        table[name] = pd.to_numeric(table[name], errors="coerce")
        lost = int(before - table[name].notna().sum())
        if lost > 0:
            unreadable[name] = lost

    return table, missing, missing_keys, unreadable


def column_has_values(table, name, missing):
    """Is this column in the file, and does it hold anything?

    Three separate things can be true of a column and only one of them
    lets a regulation read it. It can be absent. It can be a column the
    reading step added, empty, because the file did not carry it. Or it
    can be there with nothing in it, which looked like a working column
    and let a file of nothing but ID numbers through the gate.
    """
    if name not in table.columns or name in missing:
        return False
    return bool(table[name].notna().any())


def rule_tally(found):
    """What each regulation did to this whole file.

    Asks the regulations themselves, row by row, and counts the students
    each one actually caught. A rule is silent here only when no row in
    the file let it speak, and then it names the columns it was waiting
    for rather than claiming the file does not have them.
    """
    rules = university_rules.columns_the_rules_need()
    names = {r["rule"]: set() for r in rules}
    # the alert names a rule produces, mapped back to the rule itself
    produced_by = {
        "work not handed in": "work not handed in",
        "submissions out of range": None,
        "absence in person": "absence",
        "absence online": "absence",
        "cannot reach the pass mark": "marks still reachable",
        "very little margin left": "marks still reachable",
        "window closed": "the withdrawal window",
        "window closing": "the withdrawal window",
        "window open": "the withdrawal window",
    }

    spoke = {r["rule"]: False for r in rules}
    for i in range(len(found)):
        row = found.iloc[i]
        for alert in university_rules.cohort_row_summary(row)["alerts"]:
            owner = produced_by.get(alert["rule"])
            if owner is None:
                continue
            spoke[owner] = True
            if alert["level"] != "information":
                names[owner].add(str(row["id_student"]))

    out = []
    for r in rules:
        here = [c for c in r["columns"] if c in found.columns
                and found[c].notna().any()]
        wanting = [c for c in r["columns"] if c not in here]
        out.append({
            "rule": r["rule"],
            "columns": r["columns"],
            "wanting": wanting,
            "silent": not spoke[r["rule"]],
            "caught": len(names[r["rule"]]),
        })
    return out


def rules_that_can_work(table, missing):
    """Which regulations this file lets speak, and which it silences.

    The regulations name every column each of them reads and say whether
    one is enough or all are needed. A regulation that cannot read what
    it needs is not broken and is not ignored — it is silent, and it
    says so, naming the columns it was waiting for.
    """
    working, silent = [], []
    for rule in university_rules.columns_the_rules_need():
        have = [c for c in rule["columns"]
                if column_has_values(table, c, missing)]
        if rule["needs"] == "any":
            can = len(have) > 0
        else:
            can = len(have) == len(rule["columns"])
        found = dict(rule)
        found["wanting"] = [c for c in rule["columns"] if c not in have]
        (working if can else silent).append(found)
    return working, silent

def tile(number, label, colour):
    """One counting tile, the same shape the simulation uses."""
    css = "tile " + colour if colour else "tile"
    return ('<div class="' + css + '"><div class="n">' + str(number)
            + '</div><div class="t">' + label + '</div></div>')

def register_rows(people, found, kind):
    """One row per student, ready to be put on screen."""
    caught = say("by_rule") if kind == "rule" else say("by_model")
    rows = []
    for who in people:
        mine = found[found["id_student"] == who]
        bad = mine[mine["flagged"]]
        rows.append({
            say("student"): str(who),
            say("col_caught"): caught,
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

    # The colour says which half of the system caught the student: red for a
    # regulation, orange for the model. It is the same pair of colours the
    # rest of the system uses, so the register can be read at a glance.
    paint = "#fdecea;color:#b3261e" if kind == "rule" else "#fff4e8;color:#b45309"

    def colour_caught(value):
        return "background-color:" + paint + ";font-weight:600"

    def colour_state(value):
        if value == say("sent"):
            return "background-color:#eaf4ef;color:#1b8354;font-weight:600"
        return "color:#6b7280"

    return (table.style
            .map(colour_caught, subset=[say("col_caught")])
            .map(colour_state, subset=[say("state")]))

def indicators_read(row):
    """How many of the indicators carried a value for this one record."""
    seen = 0
    for name in INDICATORS:
        if pd.notna(row[name]):
            seen = seen + 1
    return seen

# ---------------------------------------------------------------
# The scan

def rule_caught(table):
    """Ask the regulations about every row of the file.

    The regulations themselves live in university_rules.py and nowhere
    else. This asks them row by row rather than repeating any of their
    arithmetic here, so there is one place where the university's rules
    are written down and one answer to what they say.
    """
    answers = []
    for i in range(len(table)):
        summary = university_rules.cohort_row_summary(table.iloc[i])
        answers.append(summary["status"] != "clear")
    return pd.Series(answers, index=table.index)

def scan(table, model, carried):
    """The whole scan, in the order the system is built on.

    carried is how many of the eleven indicators the file actually had.
    """
    found = table.copy()

    # 1. the regulations, on every row
    found["rule"] = rule_caught(found)

    # 2. the model, on the rows no regulation caught. A rule-caught row is
    #    never put to the model, so its probability stays empty.
    #
    #    A file carrying none of the eleven indicators is not put to the
    #    model at all. Asked about a row of empty cells the model answers
    #    from the pattern it learned and nothing else, so it returns the
    #    same number for every student in the file — on one test, 0.7677
    #    for all three hundred of them, which flagged the whole cohort
    #    with nothing behind any of it. The regulations still apply: they
    #    work by arithmetic on a column, and they catch who they catch.
    found["probability"] = np.nan
    left = ~found["rule"]
    if carried > 0 and left.sum() > 0:
        chances = model.predict_proba(found.loc[left, INDICATORS])[:, 1]
        found.loc[left, "probability"] = chances

    found["model_alert"] = found["probability"] >= settings.ALERT_THRESHOLD
    found["flagged"] = found["rule"] | found["model_alert"]

    # 3. how much of the picture each record actually carried
    counts = []
    for i in range(len(found)):
        counts.append(indicators_read(found.iloc[i]))
    found["indicators_read"] = counts

    return found

def columns_we_keep(table):
    """The cohort file with only the columns the system actually reads.

    A cohort file may carry far more than the system uses, including the
    six columns describing who a student is: gender, region, age band,
    deprivation band, prior education and disability. The model is never
    shown those, and database.py says plainly that nothing identifying a
    person outside the university's own records is stored.

    Keeping the uploaded file whole broke that promise quietly, because
    those columns went into the database along with everything else. So
    what is stored is the part the system read: who the record is about,
    which course, and the eleven indicators. Everything else is dropped
    before the file is written down.
    """
    # The columns the regulations read belong here too. They were left
    # out, and a file whose absence column caught three hundred students
    # was stored without it: scanning the stored copy again would have
    # found nobody, and nobody could have checked the alerts the first
    # scan sent. A record of a decision has to carry what the decision
    # was made on. The names come from the regulations themselves, so
    # this list cannot fall behind them the next time a rule changes.
    from_rules = []
    for rule in university_rules.columns_the_rules_need():
        for c in rule["columns"]:
            if c not in from_rules:
                from_rules.append(c)
    wanted = KEY_COLUMNS + INDICATORS + from_rules
    keep = []
    for c in wanted:
        if c in table.columns and c not in keep:
            keep.append(c)
    return table[keep].to_csv(index=False)

def keep_scan(file_name, table, found, carried):
    """Store the file, the run and everything the run flagged.

    The file is stored so it can be scanned again later, trimmed to the
    columns the system reads. The results go in with the reason already
    written out, so an old scan reads exactly as it read on the day rather
    than being worked out again from a model that may have changed since.
    """
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    contents = columns_we_keep(table)
    file_id = database.save_file(file_name, contents, len(found), when)

    # Every student the file carries, flagged or not, gets a row. The
    # notice table points at this one, so a notice with no student behind
    # it would be a notice about nobody. A student on two courses has one
    # row here and two records in the scan, which is the whole point of
    # keeping the two apart.
    for who, part in table.groupby("id_student"):
        credits = part["studied_credits"].max() \
            if "studied_credits" in part.columns else None
        database.save_student(
            str(who),
            None if credits is None or pd.isna(credits) else int(credits),
            when)

    people = found.groupby("id_student")["flagged"].any()
    rule_people = found[found["rule"]]["id_student"].unique()
    model_people = found[found["model_alert"]
                         & ~found["id_student"].isin(rule_people)
                         ]["id_student"].unique()

    run_id = database.save_run(
        file_id=file_id, ran_on=when, day_of_term=settings.CUT_DAY,
        threshold=settings.ALERT_THRESHOLD, students=len(people),
        records=len(found), flagged=int(people.sum()),
        by_rule=len(rule_people), by_model=len(model_people),
        indicators_carried=carried)

    rows = []
    for i in range(len(found)):
        row = found.iloc[i]
        if not row["flagged"]:
            continue
        if row["rule"]:
            said = both_languages(rule_reason, row)
        else:
            # The model's reason is the handful of indicators that pushed
            # this record towards an alert, and it is written down here
            # rather than worked out again later. A stored scan has to read
            # as it read on the day: the model may be retrained, and an old
            # notice must not quietly change its mind afterwards. Before
            # this, every model student's stored reason was the three words
            # "caught by the model", which repeat the heading above them
            # and tell an advisor nothing.
            said = both_languages(model_reason_line, row, load_model())

        rows.append({
            "student_id": str(row["id_student"]),
            "course_code": str(row["code_module"]) + " "
                           + str(row["code_presentation"]),
            "caught_by": "rule" if row["rule"] else "model",
            "reason": said["ar"],
            "reason_en": said["en"],
            "probability": (None if row["rule"]
                            else float(row["probability"])),
            "indicators_read": int(row["indicators_read"]),
        })
    database.save_results(run_id, rows)

    return run_id

def scan_log(found, missing_count, carried):
    """What the scan did, written out line by line.

    Every line is a count taken from the scan that has just run. Nothing
    here is a fixed string with a number typed into it.

    carried decides one of those lines. On a file with no indicators the
    model is not asked at all, and the log said it had been asked about
    every record and had cleared them — a sentence about work that never
    happened.
    """
    people = found.groupby("id_student")["flagged"].any()
    rule_rows = int(found["rule"].sum())
    asked = int((~found["rule"]).sum()) if carried > 0 else 0
    model_rows = int(found["model_alert"].sum())
    records = len(found)
    students = found["id_student"].nunique()

    # The first six lines count course records and the last two count
    # students, and the log says which each time. Without that, the log and
    # the tiles underneath it appear to contradict each other: a student
    # taking two courses is caught once as a student and twice as a record.
    ar = [
        "قرأ " + count_records(records) + " تخص " + count_students(students),
        "المؤشرات التي يحملها الملف: "
        + str(len(INDICATORS) - missing_count) + " من " + str(len(INDICATORS)),
        "طُبِّقت اللوائح على السجلات كلها، وعددها " + str(records),
        "أمسكت اللائحة " + count_records(rule_rows),
        ("عُرض على النموذج " + count_records(asked)) if carried > 0
        else "لم يُسأل النموذج: الملف لا يحمل أي مؤشر",
        "أنذر النموذج عند حد " + format(settings.ALERT_THRESHOLD, ".2f")
        + " على "
        + count_records(model_rows),
        "جُمِّعت السجلات في " + count_students(students),
        "المنذَر عنهم " + count_students(int(people.sum()))
        + "، ولا شيء على " + count_students(int((~people).sum())),
    ]
    en = [
        "read " + count_records(records) + " for "
        + count_students(students),
        "indicators carried by this file: "
        + str(len(INDICATORS) - missing_count) + " of " + str(len(INDICATORS)),
        "regulations applied to all " + count_records(records),
        "caught by a regulation: " + count_records(rule_rows),
        ("put to the model: " + count_records(asked)) if carried > 0
        else "the model was not asked: the file carries no indicator",
        "flagged by the model at "
        + format(settings.ALERT_THRESHOLD, ".2f") + ": "
        + count_records(model_rows),
        "gathered into " + count_students(students),
        "flagged: " + count_students(int(people.sum())) + "   clear: "
        + count_students(int((~people).sum())),
    ]

    lines = ar if st.session_state["lang"] == "ar" else en
    done = "تم" if st.session_state["lang"] == "ar" else "done"

    out = ""
    for line in lines:
        out = out + '<span class="d">&gt;</span> ' + line + "<br>"
    return out + "<b>" + done + "</b>"

def both_languages(maker, row, model=None):
    """One reason, written in Arabic and in English.

    A stored record outlives the language it was written in. The reason
    used to be saved in whichever language was on screen at the moment
    of the scan, so a register opened in Arabic showed English sentences
    from a scan run in English the week before. Both are written down,
    and the screen picks the one it needs.
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


def rule_reason(row):
    """Why a regulation caught this record, in the language on screen.

    It asks the regulations which of them spoke about this row and writes
    out what they said. It used to write the submissions rule's sentence
    whatever had actually caught the record, so a student barred for
    absence was told about work they had handed in — and on a file with
    no submissions column at all, the sentence came out full of "nan".

    The English is the regulation's own wording, taken as it stands. Only
    the Arabic is written here, because a rule should not have to know
    which language a committee is reading in.
    """
    summary = university_rules.cohort_row_summary(row)
    said = []
    for alert in summary["alerts"]:
        if alert["level"] == "information":
            continue                      # a silent rule caught nobody
        if st.session_state["lang"] != "ar":
            said.append(alert["message"])
        else:
            said.append(rule_said_in_arabic(alert, row))
    if len(said) == 0:
        return ""
    return " ".join(said)


def rule_said_in_arabic(alert, row):
    """One regulation's alert, in Arabic, with its own numbers in it."""
    name = alert["rule"]

    if name == "work not handed in":
        handed = float(row["submission_rate"])
        if handed == 0:
            return ("لم يُسلّم شيئاً من العمل المطلوب. العمل غير المسلَّم "
                    "يُرصد صفراً ما لم يُقدَّم عذر.")
        return ("سُلِّم " + str(round(handed, 1)) + "٪ من العمل المطلوب، و"
                + str(round(100 - handed, 1)) + "٪ لم يُسلَّم. غير المسلَّم "
                "يُرصد صفراً ما لم يُقدَّم عذر.")

    if name == "absence in person":
        missed = int(float(row["absence_in_person"]))
        limit = settings.MAX_ABSENCE_IN_PERSON
        if missed >= limit:
            return ("محروم من المقرر. " + str(missed) + " غيابات بغير عذر، "
                    "والحد " + str(limit) + ".")
        return ("غياب واحد يفصله عن الحرمان. " + str(missed) + " من "
                + str(limit) + ".")

    if name in ["cannot reach the pass mark", "very little margin left"]:
        # The same arithmetic the rule did, so the Arabic names the same
        # numbers the English does. Written without them, the sentence
        # said a student could not pass and never said what their ceiling
        # was — an alert with no figure in it is not a reason.
        due = float(row["due_so_far"])
        got = float(row["early_score"])
        highest = round(got / 100.0 * due + (100.0 - due), 1)
        if name == "cannot reach the pass mark":
            return ("لا يمكنه النجاح. أعلى مجموع ما زال في متناوله "
                    + str(highest) + " ودرجة النجاح "
                    + str(settings.PASS_MARK) + ".")
        return ("هامش ضيق: " + str(round(highest - settings.PASS_MARK, 1))
                + " درجة فقط فوق درجة النجاح. يحتاج درجات شبه كاملة في كل "
                "ما تبقى.")

    if name == "absence online":
        missed = int(float(row["absence_online"]))
        limit = settings.MAX_ABSENCE_ONLINE
        if missed >= limit:
            return ("محروم من المقرر. " + str(missed) + " غيابات إلكترونية "
                    "بغير عذر، والحد " + str(limit) + ".")
        return ("غياب إلكتروني واحد يفصله عن الحرمان. " + str(missed)
                + " من " + str(limit) + ".")

    # A rule this screen has no Arabic sentence for yet. Its own wording
    # is shown rather than nothing at all, so no alert goes out blank.
    return alert["message"]

def model_reasons(row, model, how_many=3):
    """Which indicators pushed this record towards an alert, and by how much.

    LightGBM gives the push of each indicator in log odds, which is not a
    number to show anybody. So each push is reported as its share of the
    total push on this record: a plain statement of which indicators carried
    the decision, without pretending the units are marks or percentages.
    """
    one = row[INDICATORS].to_frame().T.astype(float)
    pushes = model.predict(one, pred_contrib=True)[0][:len(INDICATORS)]

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

        # Each indicator is shown in the unit it is measured in. Several
        # of them are percentages and used to be printed as a bare number,
        # so a lateness of two thirds read as "66.67" with nothing to say
        # what that was two thirds of.
        if pd.isna(value):
            shown = "—"
        elif name in ["submission_rate", "late_rate", "activity_days_rate",
                      "early_score"]:
            shown = format(float(value), ".1f") + "%"
        elif name in ["late_registration", "early_score_missing",
                      "late_rate_missing"]:
            shown = say("yes") if float(value) >= 1 else say("no")
        else:
            shown = format(float(value), ".0f")

        out.append({
            "name": word(name),
            "share": round(abs(pushes[i]) / total * 100, 1),
            "value": value,
            "shown": shown,
        })
    return out

def group_words(attribute, group):
    """A name for one audited group, short enough to read under a bar.

    The audit's own labels are F, M, True, False and two age bands. Put
    on a chart as they are, half of them say nothing and the other half
    say the wrong thing, so each one is given a short plain name. They
    are kept short on purpose: a long label is turned on its edge by the
    chart and then cut off.
    """
    key = "grp_" + attribute + "_" + group
    if key in TEXT:
        return say(key)
    return group


def model_reason_line(row, model):
    """The same reason, written as one line of text.

    A stored scan and a sent notice have to carry the same justification,
    because they are two records of one decision. Writing the line in one
    place is what keeps them from drifting apart.
    """
    parts = []
    for push in model_reasons(row, model):
        parts.append(push["name"] + " " + push["shown"]
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
if "log_of" not in st.session_state:
    st.session_state["log_of"] = None
if "file_now" not in st.session_state:
    st.session_state["file_now"] = ""
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
# nothing else. An earlier version put an account name and initials
# beside it, copied from the mock up: there is no sign in and there are
# no user accounts here, so showing one would claim something untrue.
#
# The language switch used to live in a side panel that existed for it
# alone. The panel took three hundred pixels of the screen, sat on the
# wrong side in Arabic, and folding it away hid the only way back to the
# other language. It belongs here, across from the mark.
switch, mark = st.columns([1, 5])
with switch:
    if st.button(say("lang"), key="lang_switch"):
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

model = load_model()

tab_scan, tab_register, tab_evidence = st.tabs(
    [say("tab_scan"), say("tab_register"), say("tab_evidence")])

# ---------------------------------------------------------------
with tab_scan:
    # Where the cohort comes from. This used to be a radio with a pair of
    # drawn cards underneath repeating the same two choices, so the screen
    # asked the same question twice and only one of the two answered. It is
    # one control now, and what the cards carried — what each source is and
    # whether it is open to us — is written under its own option.
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
            # in UTF-8. Reading it blindly used to throw the error straight
            # onto the screen and take the rest of the page with it, so the
            # two encodings a Windows machine is likely to produce are tried
            # as well, and a file in none of them is refused politely.
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

            # A different file than last time clears the screen straight
            # away. Otherwise the name at the top would say one file while
            # the register underneath still showed another one's students.
            if st.session_state["file_now"] != handed.name + contents[:200]:
                st.session_state["file_now"] = handed.name + contents[:200]
                st.session_state["found"] = None
                st.session_state["log_of"] = None
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
            if contents != "":
                try:
                    table, missing, missing_keys, unreadable = read_cohort(
                        io.StringIO(contents))
                except Exception:
                    st.error(say("bad_file"))
                if table is not None and len(table) == 0:
                    st.error(say("empty_file"))
                    table = None

            # A file that could not be read stops the scan, and nothing
            # more. The rest of the screen, including the shelf of files
            # scanned before, stays where it is: losing the whole page
            # because one upload was the wrong format helps nobody.
            if table is not None:
                carried = len(INDICATORS) - len(missing)

                # Is this file already here? The answer comes from its
                # contents, not its name — and from the part of those
                # contents the system actually reads, which is the same
                # part that gets stored. Fingerprinting the whole upload
                # instead would mean an uploaded file never matched what
                # was saved from it.
                kept = columns_we_keep(table)
                already = database.find_file(kept)
                newer_than = database.other_versions(handed.name, kept)

                if already is not None:
                    st.info(say("file_known",
                                when=plain_date(str(already["uploaded_on"])),
                                rows=int(already["rows_in_file"])))
                elif len(newer_than) > 0:
                    st.warning(say("file_updated", n=len(newer_than)))

                st.write("**" + say("cols_read") + ":** "
                         + str(carried) + " / " + str(len(INDICATORS)))

                working, silent = rules_that_can_work(table, missing)

                # Everything this file does not carry, said once, in one
                # place. An advisor reading a message that names the first
                # missing thing and stops has to fix it, upload again, and
                # be told the second. All of it goes in together.
                gaps = []
                if len(missing_keys) > 0:
                    gaps.append(block(say("gap_keys"),
                                      [word(n) + "  `" + n + "`"
                                       for n in missing_keys]))
                if len(missing) > 0:
                    gaps.append(block(
                        say("gap_cols") + "  (" + str(len(missing)) + " / "
                        + str(len(INDICATORS)) + ")",
                        [word(n) + "  `" + n + "`" for n in missing],
                        say("cols_missing")))
                if len(unreadable) > 0:
                    gaps.append(block(
                        say("gap_text"),
                        [word(n) + "  `" + n + "`  —  "
                         + count_cells(unreadable[n])
                         for n in unreadable],
                        say("gap_text_note")))
                if len(silent) > 0:
                    gaps.append(block(say("gap_rules"),
                                      [rule_name(r["rule"]) + "  "
                                       + " ".join("`" + c + "`"
                                                  for c in r["wanting"])
                                       for r in silent]))

                # The scan stops for one of two reasons, and only these.
                # Without a student number there is nobody to name. With
                # no indicator and no regulation there is nothing to judge
                # on: the model, handed a row of empty cells, returns the
                # same number for every student in the file and would
                # warn about the whole cohort for no stated reason.
                catchers = [r for r in working if r["catches"]]
                stop = len(missing_keys) > 0 or (carried == 0
                                                 and len(catchers) == 0)
                if stop:
                    gaps.insert(0, "#### " + say("cannot_scan") + "\n\n"
                                + say("why_stopped"))

                if len(gaps) > 0:
                    (st.error if stop else st.warning)("\n\n".join(gaps))

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
                        found = scan(table, model, carried)
                        st.session_state["found"] = found
                        st.session_state["log_of"] = (len(missing), carried)
                        st.session_state["run_id"] = keep_scan(
                            handed.name, table, found, carried)
                    st.rerun()

    found = st.session_state["found"]
    if found is not None:
        people = found.groupby("id_student")["flagged"].any()
        rule_people = found[found["rule"]]["id_student"].unique()
        model_people = found[found["model_alert"]
                             & ~found["id_student"].isin(rule_people)
                             ]["id_student"].unique()

        # the five steps, all of them done by the time this is drawn
        steps = ""
        for number, key in enumerate(["step1", "step2", "step3", "step4",
                                      "step5"], start=1):
            steps = steps + ('<div class="sp2 done"><span class="dot">'
                             + str(number) + '</span>'
                             '<span class="tx">' + say(key) + '</span></div>')
        st.markdown('<div class="steps">' + steps + '</div>',
                    unsafe_allow_html=True)

        # Written out now, in the language on screen now. It used to be
        # rendered once at scan time and printed unchanged ever after, so
        # switching language left eight English lines under an Arabic
        # heading. Only the two numbers it needs are kept.
        if st.session_state["log_of"] is not None:
            missed_count, carried_then = st.session_state["log_of"]
            st.markdown('<div class="log">'
                        + scan_log(found, missed_count, carried_then)
                        + '</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="split">'
            + tile(len(people), say("students"), "")
            + tile(int(people.sum()), say("flagged"), "o")
            + tile(len(rule_people), say("by_rule"), "r")
            + tile(len(model_people), say("by_model"), "o")
            + tile(int((~people).sum()), say("clear"), "g")
            + '</div>', unsafe_allow_html=True)
        st.caption(count_records(len(found)))

        # What the regulations could and could not reach in this file. Each
        # line is the rule's own answer, not a claim made on its behalf.
        st.divider()
        st.subheader(say("rules_state"))

        # What each regulation did to THIS file, counted over the whole
        # file. It used to be read off the first row alone, so one blank
        # cell at the top made the table announce that a column was not
        # in the file while the register below listed ninety four
        # students that same rule had caught. And every rule that spoke
        # was credited with the same number — the students caught by any
        # rule at all — so three rules each claimed the same three.
        lines = []
        for r in rule_tally(found):
            if r["silent"]:
                verdict = say("silent")
                if len(r["wanting"]) > 0:
                    verdict = verdict + " — " + joiner().join(r["wanting"])
            else:
                verdict = say("speaks") + " — " + (
                    say("caught_nobody") if r["caught"] == 0
                    else count_students(r["caught"]))
            lines.append({
                say("rule_name"): rule_name(r["rule"]),
                say("rule_needs"): "، ".join(r["columns"])
                if st.session_state["lang"] == "ar"
                else ", ".join(r["columns"]),
                say("rule_says"): verdict,
            })
        st.dataframe(pd.DataFrame(lines), hide_index=True, width="stretch")

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
            say("col_by_rule"): runs["by_rule"],
            say("col_by_model"): runs["by_model"],
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
                table, missing, _, _ = read_cohort(io.StringIO(contents))
                stored_carried = len(INDICATORS) - len(missing)
                with st.spinner(say("scanning")):
                    found = scan(table, model, stored_carried)
                    st.session_state["found"] = found
                    st.session_state["stored"] = None
                    st.session_state["log_of"] = (len(missing),
                                                  stored_carried)
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

        # A student belongs to one block, not two. If any one of his course
        # records was caught by a regulation, that is why the committee is
        # looking at him, even where another of his courses was flagged by
        # the model instead. This is the same order the live register uses,
        # and the two have to agree or the same scan reads two ways.
        rule_students = stored[stored["caught_by"] == "rule"]["student_id"]

        for kind, heading in [("rule", say("rule_block")),
                              ("model", say("model_block"))]:
            if kind == "rule":
                part = stored[stored["student_id"].isin(rule_students)]
            else:
                part = stored[~stored["student_id"].isin(rule_students)]
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
            # used to give a student count over it.
            st.markdown('<div class="blockhead">' + heading
                        + ' <span class="sm">· '
                        + count_students(part["student_id"].nunique())
                        + " · " + count_records(len(part))
                        + '</span></div>',
                        unsafe_allow_html=True)

            # A probability that does not exist is left as an empty cell
            # rather than printed as the word None. The live register says
            # in a sentence why a rule-caught student has no probability;
            # in a table an empty cell says the same thing.
            chance = []
            for v in part["probability"].values:
                chance.append("" if pd.isna(v) else format(float(v), ".2f"))

            shown = pd.DataFrame({
                say("student"): part["student_id"].values,
                say("col_course"): part["course_code"].values,
                say("col_caught"): [say("by_rule") if c == "rule"
                                    else say("by_model")
                                    for c in part["caught_by"].values],
                say("col_reason"): stored_reason(part),
                say("col_prob"): chance,
                say("col_read"): part["indicators_read"].values,
            })
            st.dataframe(shown, hide_index=True, width="stretch")

    elif found is None:
        st.caption(say("no_file"))
    else:
        rule_people = found[found["rule"]]["id_student"].unique()
        model_people = found[found["model_alert"]
                             & ~found["id_student"].isin(rule_people)
                             ]["id_student"].unique()

        # Which block to show, and a search box. With 155 students on the
        # register, an advisor looking for one of them needs both.
        left, right = st.columns([2, 3])
        with left:
            which = st.radio(
                say("source"),
                [say("chip_all"), say("by_rule"), say("by_model")],
                horizontal=True, label_visibility="collapsed")
        with right:
            looking_for = st.text_input(say("search"), "",
                                        label_visibility="collapsed",
                                        placeholder=say("search"))

        blocks = []
        if which in (say("chip_all"), say("by_rule")):
            blocks.append(("rule", rule_people, say("rule_block")))
        if which in (say("chip_all"), say("by_model")):
            blocks.append(("model", model_people, say("model_block")))

        picked = None
        for kind, people, heading in blocks:
            shown = [w for w in people
                     if looking_for.strip() == ""
                     or looking_for.strip() in str(w)]
            if len(shown) == 0:
                continue

            st.markdown('<div class="blockhead">' + heading
                        + ' <span class="sm">· ' + count_students(len(shown))
                        + '</span></div>',
                        unsafe_allow_html=True)

            table = register_rows(shown, found, kind)
            chosen = st.dataframe(
                table, hide_index=True, width="stretch",
                on_select="rerun", selection_mode="single-row",
                # The search text is part of the key on purpose. Changing
                # the search builds a fresh table with nothing selected,
                # instead of carrying the old row's position over to a
                # different list of students.
                key="reg_" + kind + "_" + which + "_" + looking_for.strip())

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
            everyone = list(rule_people) + list(model_people)
            if not any(looking_for.strip() in str(w) for w in everyone):
                st.caption(say("nobody"))

        # A cohort where nobody is flagged used to leave this part of the
        # screen blank, which reads as a system that did not run rather
        # than one that ran and found nothing.
        if len(rule_people) == 0 and len(model_people) == 0:
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

                inside = ""
                if row["rule"]:
                    inside = ('<p class="why">' + rule_reason(row) + '</p>'
                              '<p class="note">' + say("no_prob") + '</p>')
                else:
                    # The threshold is given beside the probability. A
                    # number on its own invites the question "compared to
                    # what", and the answer is the only thing that makes
                    # the number mean anything.
                    inside = ('<p class="why">' + say("prob") + ": <b>"
                              + format(float(row["probability"]), ".2f")
                              + "</b> " + say("against_threshold",
                                              t=format(settings.ALERT_THRESHOLD,
                                                       ".2f"))
                              + "</p>")
                    pushes = model_reasons(row, model)
                    # Three numbers in a row mean nothing without saying
                    # what each column is. A share of 23.9 per cent was
                    # sitting beside a value of 63.5 per cent with no word
                    # anywhere on the card to tell the two apart.
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
                    # And what the three of them add up to, so nobody has
                    # to wonder what happened to the rest of the decision.
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
                    + ' · <span class="tag '
                    + ("r" if row["rule"] else "o") + '">'
                    + (say("by_rule") if row["rule"] else say("by_model"))
                    + '</span></div>' + inside + '</div>',
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
                            caught_by="rule" if row["rule"] else "model",
                            reason=(rule_reason(row) if row["rule"]
                                    else model_reason_line(row, model)),
                            probability=(None if row["rule"]
                                         else float(row["probability"])),
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
        "Test_rows": ("سجلات الاختبار", "Test rows"),
    }
    CELLS = {
        "random (final test)": ("عشوائي — الاختبار النهائي",
                                "random (final test)"),
        "time based (newest term)": ("زمني — أحدث فصل",
                                     "time based (newest term)"),
        "the rule alone": ("اللائحة وحدها", "the rule alone"),
        "the model alone": ("النموذج وحده", "the model alone"),
        "the rule first, then the model": ("اللائحة أولاً ثم النموذج",
                                           "the rule first, then the model"),
        "who the student is": ("من هو الطالب", "who the student is"),
        "what the record says": ("ما يسجّله المقرر", "what the record says"),
        "what the student does": ("ما يفعله الطالب", "what the student does"),
        "the record and the doing": ("السجل والسلوك معاً",
                                     "the record and the doing"),
        # the Note column of results_indicator_groups.csv
        "refused by the system, trained here only to price the refusal":
            ("يرفضها النظام، ودُرِّبت هنا فقط لقياس ثمن رفضها",
             "refused by the system, trained here only to price the refusal"),
        "what any registry already holds":
            ("ما تملكه أي إدارة قبول وتسجيل",
             "what any registry already holds"),
        "needs a platform that watches":
            ("يحتاج منصة تعليمية تسجّل النشاط",
             "needs a platform that watches"),
        "the system as built":
            ("النظام كما هو مبني", "the system as built"),
        # The three columns the fairness audit reads, and their values.
        # The published data writes them as gender / F / True; on screen
        # they are written the way a committee would say them.
        "gender": ("الجنس", "gender"),
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
        # The threshold is written 0.30 on the student's card and in the
        # scan log, so it is written 0.30 here too. One number should not
        # have two spellings inside one system.
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

    show("results_final.csv", say("ev_test"))
    show("results_rule_vs_model.csv", say("ev_rule"))

    # The indicator groups carry a long sentence in their last column, so
    # the table is given room rather than clipping the one line that
    # explains why the system refuses to look at who a student is.
    show("results_indicator_groups.csv", say("ev_groups"), height=220)

    # The same comparison run at day fifty as well. A committee that asks
    # why the scan waits until day one hundred is asking what the earlier
    # day looked like, and the answer is a table, not an opinion.
    st.caption(say("ev_timing_note"))
    show("results_timing.csv", say("ev_timing"), height=250)

    st.divider()
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

    st.divider()
    show("results_model_comparison.csv", say("ev_models"))

    st.divider()
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
        bars = drawn.set_index(say("col_group"))[["Recall", "Precision"]]
        bars.columns = [say("m_recall"), say("m_precision")]
        st.bar_chart(bars, color=["#1b8354", "#b45309"], height=360,
                     stack=False, horizontal=True)

    st.divider()
    # The weight of each indicator, as a table with the bar drawn inside
    # it. This used to be a table of raw column names beside a separate
    # chart, and the chart cut every name off after a few letters because
    # the words are phrases, not labels. With the bar in the row, the name
    # has the whole width of a column and the bar still shows the order at
    # a glance.
    try:
        weight = pd.read_csv(settings.project_file(
            "results_feature_importance.csv"))
    except FileNotFoundError:
        weight = None

    if weight is not None:
        st.caption(say("ev_weight"))
        st.caption(say("chart_weight"))
        widest = float(weight["Share_%"].max())
        bars = ('<div class="push hd"><span>' + say("col_indicator")
                + '</span><span class="s">' + say("col_share")
                + '</span></div>')
        for i in range(len(weight)):
            share = float(weight["Share_%"].iloc[i])
            bars = bars + (
                '<div class="wrow"><span class="nm">'
                + word(str(weight["Feature"].iloc[i])) + '</span>'
                '<span class="track"><span class="fill" style="width:'
                + format(share / widest * 100, ".1f") + '%"></span></span>'
                '<span class="s">' + format(share, ".2f") + '%</span></div>')
        st.markdown('<div class="box">' + bars + '</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.caption(say("ev_data_note"))
    show("results_dataset_comparison.csv", say("ev_data"))
