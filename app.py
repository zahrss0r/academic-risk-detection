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


INDICATORS = settings.MODEL_INDICATORS

# Plain words for each indicator, in both languages. The model's column names
# are for the code; these are what a person reads.
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
}

TEXT = {
    "title":        ("نظام الكشف المبكر عن التعثر الأكاديمي",
                     "Smart Academic Risk Detection System"),
    "subtitle":     ("يقرأ الدفعة، يطبّق اللائحة، ثم ينذر المرشد",
                     "Reads the cohort, applies the regulations, notifies the advisor"),
    "tab_scan":     ("فحص الدفعة", "Scan"),
    "tab_register": ("سجل الطلاب", "Register"),
    "tab_evidence": ("أداء النظام", "Evidence"),
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
    "cols_missing": ("أعمدة غير موجودة في الملف، والنموذج أُبلغ أنها مجهولة "
                     "ولم تُعطَ صفراً",
                     "Columns this file does not carry. The model was told "
                     "they are unknown, not given a zero."),
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
    "prob":         ("احتمال التعثر", "probability"),
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
                     "correct, and catches nobody at this day"),
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
                     "والمجموع ٣٧٨٢ لأن الفحص على مجموعة التحقق لا الاختبار.",
                     "These columns are not given to the model. They are used "
                     "only to audit it. The groups total 3,782 because this "
                     "audit runs on the validation set, not the test set."),
    "ev_weight":    ("وزن كل مؤشر في القرار", "What each indicator carries"),
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
    "nobody":       ("ما فيه طالب بهذا الرقم", "No student with that number"),
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
}


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
.push .v{font-variant-numeric:tabular-nums;color:var(--ink-2)}
.push .s{font-weight:700;font-variant-numeric:tabular-nums;color:var(--orange)}
.note{font-size:13px;color:var(--muted);line-height:1.7;margin-top:12px}
/* the bar across the very top, with the university's mark on it */
.nav{background:var(--paper);border:1px solid var(--line);border-radius:12px;
     padding:12px 18px;display:flex;align-items:center;gap:16px;margin-bottom:14px}
.nav .logo img{height:34px;width:auto;display:block}
.nav .who{margin-inline-start:auto;display:flex;align-items:center;gap:10px}
.nav .av{width:34px;height:34px;border-radius:50%;background:var(--green-l);
         color:var(--green);display:flex;align-items:center;justify-content:center;
         font-size:13px;font-weight:700;flex:0 0 34px}
.nav .t1{font-size:13.5px;font-weight:600;line-height:1.3}
.nav .t2{font-size:12px;color:var(--muted);line-height:1.3}
/* the two source cards */
.src{display:flex;align-items:center;gap:12px;border:1px solid var(--line);
     background:var(--paper);border-radius:10px;padding:14px 16px;margin-bottom:9px}
.src.on{border-color:var(--green);background:var(--green-l)}
.src .rd{width:16px;height:16px;border-radius:50%;border:2px solid #c3cbd9;flex:0 0 16px}
.src.on .rd{border-color:var(--green);background:radial-gradient(var(--green) 42%,#fff 46%)}
.src .t1{font-size:14.5px;font-weight:600}
.src .t2{font-size:12.5px;color:var(--muted)}
.src .rt{margin-inline-start:auto;font-size:12.5px;color:var(--muted)}
.src.off{opacity:.55}
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
     margin-bottom:18px;direction:ltr;text-align:left}
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


def word(name):
    """The plain name of one indicator."""
    pair = WORDS.get(name, (name, name))
    return pair[0] if st.session_state["lang"] == "ar" else pair[1]


# ---------------------------------------------------------------
# Reading

@st.cache_resource
def load_model():
    return joblib.load(settings.project_file("final_model.pkl"))


def read_cohort(source):
    """Read a cohort file and line its columns up with what the model needs.

    A file may not carry every indicator. Any it does not carry is added as
    an empty column, not a zero. The model was trained to read an empty cell
    as "not known" and to decide which way to send it, so telling it the
    truth is better than filling the gap with a number nobody measured.
    """
    table = pd.read_csv(source)

    missing = []
    for name in INDICATORS:
        if name not in table.columns:
            table[name] = np.nan
            missing.append(name)

    return table, missing


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
            "": caught,
            say("failing_in"): len(bad),
            say("courses"): len(mine),
            say("state"): (say("sent") if who in st.session_state["sent"]
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
            .map(colour_caught, subset=[""])
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
    """Apply the regulations we can apply to this file.

    Rule three, work not handed in: the regulation says work that was due and
    not submitted scores zero unless an excuse is filed. A student who has
    not handed in everything that was due is exactly a student whose
    submission_rate is below 100, so the rule reads that column.

    The other three regulations are built in university_rules.py and are
    silent here, each for a reason that is written down:
      - absence         : this data has no absence column
      - marks reachable : counting the final exam's marks, no student is
                          arithmetically lost this early in the term
      - the window      : it needs the date of that course's final, which is
                          not in a cohort file
    A silent rule is not a broken rule. It is waiting for a column.
    """
    if "submission_rate" not in table.columns:
        return pd.Series(False, index=table.index)
    return table["submission_rate"] < 100


def scan(table, model):
    """The whole scan, in the order the system is built on."""
    found = table.copy()

    # 1. the regulations, on every row
    found["rule"] = rule_caught(found)

    # 2. the model, on the rows no regulation caught. A rule-caught row is
    #    never put to the model, so its probability stays empty.
    found["probability"] = np.nan
    left = ~found["rule"]
    if left.sum() > 0:
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


def keep_scan(file_name, contents, found, carried):
    """Store the file, the run and everything the run flagged.

    The file goes in whole, so it can be scanned again later. The results
    go in with the reason already written out, so an old scan reads exactly
    as it read on the day rather than being worked out again.
    """
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    file_id = database.save_file(file_name, contents, len(found), when)

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
        rows.append({
            "student_id": str(row["id_student"]),
            "course_code": str(row["code_module"]) + " "
                           + str(row["code_presentation"]),
            "caught_by": "rule" if row["rule"] else "model",
            "reason": (rule_reason(row) if row["rule"]
                       else say("by_model")),
            "probability": (None if row["rule"]
                            else float(row["probability"])),
            "indicators_read": int(row["indicators_read"]),
        })
    database.save_results(run_id, rows)

    return run_id


def scan_log(found, missing_count):
    """What the scan did, written out line by line.

    Every line is a count taken from the scan that has just run. Nothing
    here is a fixed string with a number typed into it.
    """
    people = found.groupby("id_student")["flagged"].any()
    rule_rows = int(found["rule"].sum())
    asked = int((~found["rule"]).sum())
    model_rows = int(found["model_alert"].sum())

    lines = [
        ("read " + str(len(found)) + " course records for "
         + str(found["id_student"].nunique()) + " students"),
        ("indicators carried by this file: "
         + str(len(INDICATORS) - missing_count) + " of " + str(len(INDICATORS))),
        ("regulations applied to all " + str(len(found)) + " records"),
        ("caught by a regulation: " + str(rule_rows) + " records"),
        ("put to the model: " + str(asked) + " records"),
        ("flagged by the model at " + str(settings.ALERT_THRESHOLD) + ": "
         + str(model_rows) + " records"),
        ("gathered into " + str(len(people)) + " students"),
        ("flagged: " + str(int(people.sum())) + "   clear: "
         + str(int((~people).sum()))),
    ]

    out = ""
    for line in lines:
        out = out + '<span class="d">&gt;</span> ' + line + "<br>"
    return out + '<b>done</b>'


def rule_reason(row):
    """Why a regulation caught this record. Computed from the row itself."""
    handed_in = row["submission_rate"]
    not_handed = round(100 - handed_in, 1)
    if st.session_state["lang"] == "ar":
        if handed_in == 0:
            return ("لم يُسلّم شيئاً من العمل المطلوب. العمل غير المسلَّم "
                    "يُرصد صفراً ما لم يُقدَّم عذر.")
        return ("سُلِّم " + str(round(handed_in, 1)) + "٪ من العمل المطلوب، "
                "و" + str(not_handed) + "٪ لم يُسلَّم. غير المسلَّم يُرصد "
                "صفراً ما لم يُقدَّم عذر.")
    if handed_in == 0:
        return ("Nothing was handed in of the work that was due. Work not "
                "submitted scores zero unless an excuse is filed.")
    return (str(round(handed_in, 1)) + "% of the work that was due was handed "
            "in, and " + str(not_handed) + "% was not. What was not handed in "
            "scores zero unless an excuse is filed.")


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
        out.append({
            "name": word(INDICATORS[i]),
            "share": round(abs(pushes[i]) / total * 100, 1),
            "value": row[INDICATORS[i]],
        })
    return out


# ---------------------------------------------------------------
# The page

st.set_page_config(page_title="Academic Risk Detection", layout="wide")

if "lang" not in st.session_state:
    st.session_state["lang"] = "ar"
if "sent" not in st.session_state:
    st.session_state["sent"] = {}
if "found" not in st.session_state:
    st.session_state["found"] = None
if "log" not in st.session_state:
    st.session_state["log"] = ""
if "file_now" not in st.session_state:
    st.session_state["file_now"] = ""
if "run_id" not in st.session_state:
    st.session_state["run_id"] = None
if "stored" not in st.session_state:
    st.session_state["stored"] = None
if "stored_when" not in st.session_state:
    st.session_state["stored_when"] = ""

with st.sidebar:
    if st.button(say("lang")):
        st.session_state["lang"] = "en" if st.session_state["lang"] == "ar" else "ar"
        st.rerun()

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


# The bar carries the university's mark and nothing else. An earlier
# version put an account name and initials beside it, copied from the
# mock up. There is no sign in and there are no user accounts in this
# system, so showing one would be claiming something that is not there.
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
database.create_tables()

tab_scan, tab_register, tab_evidence = st.tabs(
    [say("tab_scan"), say("tab_register"), say("tab_evidence")])


# ---------------------------------------------------------------
with tab_scan:
    source = st.radio(say("source"), [say("src_file"), say("src_db")],
                      horizontal=True, label_visibility="collapsed")

    # The two sources drawn as cards, the way the simulation drew them. The
    # radio above is what Streamlit listens to; these show which one is on.
    on_file = source == say("src_file")
    st.markdown(
        '<div class="src ' + ("on" if on_file else "") + '">'
        '<span class="rd"></span><span><span class="t1">' + say("src_file")
        + '</span><br><span class="t2">' + say("src_file_sub") + '</span></span>'
        '<span class="rt">' + say("ready") + '</span></div>'
        '<div class="src off ' + ("on" if not on_file else "") + '">'
        '<span class="rd"></span><span><span class="t1">' + say("src_db")
        + '</span><br><span class="t2">' + say("src_db_sub") + '</span></span>'
        '<span class="rt">' + say("not_ready") + '</span></div>',
        unsafe_allow_html=True)

    if not on_file:
        st.info(say("db_closed"))

    else:
        handed = st.file_uploader(say("upload"), type=["csv"])

        # Streamlit's uploader takes a dropped file as well as a chosen one,
        # but says so only in English and only on the button. This line says
        # it in the language the rest of the screen is in.
        st.caption(say("drag"))

        if handed is None:
            st.caption(say("no_file"))
        else:
            contents = handed.getvalue().decode("utf-8")

            # A different file than last time clears the screen straight
            # away. Otherwise the name at the top would say one file while
            # the register underneath still showed another one's students.
            if st.session_state["file_now"] != handed.name + contents[:200]:
                st.session_state["file_now"] = handed.name + contents[:200]
                st.session_state["found"] = None
                st.session_state["log"] = ""
                st.session_state["sent"] = {}

            table, missing = read_cohort(io.StringIO(contents))
            carried = len(INDICATORS) - len(missing)

            # Is this file already here? The answer comes from its contents,
            # not its name.
            already = database.find_file(contents)
            newer_than = database.other_versions(handed.name, contents)

            if already is not None:
                st.info(say("file_known",
                            when=str(already["uploaded_on"]),
                            rows=int(already["rows_in_file"])))
            elif len(newer_than) > 0:
                st.warning(say("file_updated", n=len(newer_than)))

            st.write("**" + say("cols_read") + ":** "
                     + str(carried) + " / " + str(len(INDICATORS)))
            if len(missing) > 0:
                st.warning(say("cols_missing") + ": "
                           + ", ".join([word(n) for n in missing]))

            if st.button(say("run"), type="primary"):
                with st.spinner(say("scanning")):
                    found = scan(table, model)
                    st.session_state["found"] = found
                    st.session_state["log"] = scan_log(found, len(missing))
                    st.session_state["run_id"] = keep_scan(
                        handed.name, contents, found, carried)
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

        st.markdown('<div class="log">' + st.session_state["log"] + '</div>',
                    unsafe_allow_html=True)

        st.markdown(
            '<div class="split">'
            + tile(len(people), say("students"), "")
            + tile(int(people.sum()), say("flagged"), "o")
            + tile(len(rule_people), say("by_rule"), "r")
            + tile(len(model_people), say("by_model"), "o")
            + tile(int((~people).sum()), say("clear"), "g")
            + '</div>', unsafe_allow_html=True)
        st.caption(str(len(found)) + " " + say("rows"))

        # What the regulations could and could not reach in this file. Each
        # line is the rule's own answer, not a claim made on its behalf.
        st.divider()
        st.subheader(say("rules_state"))
        st.dataframe(pd.DataFrame([
            {say("rule_name"): "work not handed in",
             say("rule_needs"): "submission_rate",
             say("rule_says"): say("speaks") + " — " + str(len(rule_people))
                               + " " + say("students")},
            {say("rule_name"): "marks still reachable",
             say("rule_needs"): "assessment weights",
             say("rule_says"): say("silent_ok")},
            {say("rule_name"): "withdrawal window",
             say("rule_needs"): "exam_day",
             say("rule_says"): say("silent")},
            {say("rule_name"): "absence",
             say("rule_needs"): "absence count",
             say("rule_says"): say("silent")},
        ]), hide_index=True, width="stretch")

    # Everything scanned so far, so any earlier cohort can be opened again
    # or put through the system a second time.
    st.divider()
    st.subheader(say("stored"))

    runs = database.get_runs()
    if len(runs) == 0:
        st.caption(say("no_stored"))
    else:
        shelf = pd.DataFrame({
            say("src_file"): runs["file_name"],
            say("pick_run"): runs["ran_on"],
            say("students"): runs["students"],
            say("flagged"): runs["flagged"],
            say("by_rule"): runs["by_rule"],
            say("by_model"): runs["by_model"],
            say("cols_read"): runs["indicators_carried"],
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
                table, missing = read_cohort(io.StringIO(contents))
                with st.spinner(say("scanning")):
                    found = scan(table, model)
                    st.session_state["found"] = found
                    st.session_state["stored"] = None
                    st.session_state["log"] = scan_log(found, len(missing))
                    st.session_state["run_id"] = keep_scan(
                        str(run["file_name"]), contents, found,
                        len(INDICATORS) - len(missing))
                st.rerun()


# ---------------------------------------------------------------
with tab_register:
    found = st.session_state["found"]
    stored = st.session_state["stored"]

    if found is None and stored is not None:
        # A result opened from the database. It is read back exactly as it
        # was written, so there is no model here and nothing is worked out
        # again: these are the sentences the system gave on the day.
        st.info(say("from_store", when=st.session_state["stored_when"]))

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
            st.markdown('<div class="blockhead">' + heading
                        + ' <span class="sm">· '
                        + str(part["student_id"].nunique()) + " "
                        + say("students") + '</span></div>',
                        unsafe_allow_html=True)
            shown = pd.DataFrame({
                say("student"): part["student_id"].values,
                say("courses"): part["course_code"].values,
                say("why"): part["reason"].values,
                say("prob"): part["probability"].values,
                say("cols_read"): part["indicators_read"].values,
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
                        + ' <span class="sm">· ' + str(len(shown)) + " "
                        + say("students") + '</span></div>',
                        unsafe_allow_html=True)

            table = register_rows(shown, found, kind)
            chosen = st.dataframe(
                table, hide_index=True, width="stretch",
                on_select="rerun", selection_mode="single-row",
                key="reg_" + kind)

            rows_picked = chosen["selection"]["rows"]
            if len(rows_picked) > 0:
                picked = shown[rows_picked[0]]

        if picked is None and looking_for.strip() != "":
            everyone = list(rule_people) + list(model_people)
            if not any(looking_for.strip() in str(w) for w in everyone):
                st.caption(say("nobody"))

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
                    inside = ('<p class="why">' + say("prob") + ": <b>"
                              + str(round(float(row["probability"]), 4))
                              + "</b></p>")
                    for push in model_reasons(row, model):
                        value = push["value"]
                        shown = "—" if pd.isna(value) else str(round(float(value), 2))
                        inside = inside + (
                            '<div class="push"><span>' + push["name"] + '</span>'
                            '<span class="v">' + shown + '</span>'
                            '<span class="s">' + str(push["share"]) + '%</span>'
                            '</div>')

                st.markdown(
                    '<div class="box">'
                    '<h3>' + str(row["code_module"]) + " "
                    + str(row["code_presentation"]) + '</h3>'
                    '<div class="cap">'
                    + say("read_n", n=int(row["indicators_read"]),
                          total=len(INDICATORS))
                    + ' · <span class="tag '
                    + ("r" if row["rule"] else "o") + '">'
                    + (say("by_rule") if row["rule"] else say("by_model"))
                    + '</span></div>' + inside + '</div>',
                    unsafe_allow_html=True)

            if who in st.session_state["sent"]:
                st.success(say("sent_at") + ": " + st.session_state["sent"][who])
            else:
                if st.button(say("send"), key="send_" + str(who), type="primary"):
                    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state["sent"][who] = when
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
                                    else "model, p="
                                         + str(round(float(row["probability"]), 4))),
                            probability=(None if row["rule"]
                                         else float(row["probability"])),
                            indicators_read=int(row["indicators_read"]),
                            state="sent")
                    st.rerun()
            st.caption(say("stop_note"))


# ---------------------------------------------------------------
with tab_evidence:
    def show(file_name, caption):
        try:
            st.caption(caption)
            st.dataframe(pd.read_csv(settings.project_file(file_name)),
                         hide_index=True, width="stretch")
        except FileNotFoundError:
            st.caption(caption + " — " + file_name)

    show("results_final.csv", say("ev_test"))
    show("results_rule_vs_model.csv", say("ev_rule"))
    show("results_indicator_groups.csv", say("ev_groups"))
    show("results_threshold_table.csv", say("ev_thresh"))
    show("results_model_comparison.csv", say("ev_models"))

    st.divider()
    st.caption(say("ev_fair_note"))
    show("results_fairness.csv", say("ev_fair"))

    st.divider()
    show("results_feature_importance.csv", say("ev_weight"))
