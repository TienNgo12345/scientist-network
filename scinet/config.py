"""Cấu hình chung cho pipeline xây dựng mạng các nhà khoa học quốc tế."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # wikitext thô (JSONL)
INTERIM = DATA / "interim"    # kết quả trung gian
OUT = DATA / "output"         # nodes.csv, edges.csv
REPORTS = ROOT / "reports"    # hình, bảng thống kê
for p in (RAW, INTERIM, OUT, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SNA-ScientistNetwork/1.0 (student course project; contact via GitHub repo)"
REQUEST_DELAY = 0.2          # giây giữa 2 request (lịch sự với API)
BATCH = 50                   # API cho phép tối đa 50 title / request

# Mục tiêu kích thước mạng
TARGET_SCIENTISTS = 3200     # >= 2000 nút nhà khoa học
MAX_BFS_DEPTH = 3

# Các category hạt giống (seed). Có thể bổ sung/bớt.
SEED_CATEGORIES = [
    "Category:Nobel laureates in Physics",
    "Category:Nobel laureates in Chemistry",
    "Category:Nobel laureates in Physiology or Medicine",
    "Category:Nobel laureates in Economics",
    "Category:Fields Medalists",
    "Category:Turing Award laureates",
    "Category:Wolf Prize in Physics laureates",
    "Category:Wolf Prize in Mathematics laureates",
    "Category:Abel Prize laureates",
    "Category:Recipients of the Copley Medal",
    "Category:Royal Medal winners",
    "Category:Kyoto laureates in Basic Sciences",
    "Category:Kyoto laureates in Advanced Technology",
]

# Template infobox được coi là "nhà khoa học"
SCIENTIST_INFOBOXES = {
    "infobox scientist", "infobox academic", "infobox mathematician",
    "infobox physicist", "infobox chemist", "infobox economist",
    "infobox engineer", "infobox medical person", "infobox biologist",
    "infobox astronomer", "infobox computer scientist", "infobox philosopher",
}
# Template infobox được coi là tổ chức/viện nghiên cứu
INSTITUTION_INFOBOXES = {
    "infobox university", "infobox organization", "infobox laboratory",
    "infobox research institute", "infobox company", "infobox government agency",
    "infobox school", "infobox college", "infobox academy", "infobox institute",
    "infobox scientific organization", "infobox learned society",
}

# Ánh xạ trường Infobox -> (loại quan hệ, hướng, loại nút đích)
# hướng "out": chủ thể -> đích ; "in": đích -> chủ thể
INFOBOX_FIELD_MAP = {
    "doctoral_advisor":   ("DOCTORAL_ADVISOR_OF", "in", "Scientist"),
    "doctoral_advisors":  ("DOCTORAL_ADVISOR_OF", "in", "Scientist"),
    "doctoral_students":  ("DOCTORAL_ADVISOR_OF", "out", "Scientist"),
    "academic_advisors":  ("ACADEMIC_ADVISOR_OF", "in", "Scientist"),
    "academic_advisor":   ("ACADEMIC_ADVISOR_OF", "in", "Scientist"),
    "notable_students":   ("ACADEMIC_ADVISOR_OF", "out", "Scientist"),
    "other_students":     ("ACADEMIC_ADVISOR_OF", "out", "Scientist"),
    "influences":         ("INFLUENCED", "in", "Scientist"),
    "influenced":         ("INFLUENCED", "out", "Scientist"),
    "workplaces":         ("WORKED_AT", "out", "Institution"),
    "work_institutions":  ("WORKED_AT", "out", "Institution"),
    "institutions":       ("WORKED_AT", "out", "Institution"),
    "institution":        ("WORKED_AT", "out", "Institution"),
    "alma_mater":         ("STUDIED_AT", "out", "Institution"),
    "education":          ("STUDIED_AT", "out", "Institution"),
    "fields":             ("WORKS_IN_FIELD", "out", "Field"),
    "field":              ("WORKS_IN_FIELD", "out", "Field"),
    "discipline":         ("WORKS_IN_FIELD", "out", "Field"),
    "awards":             ("RECEIVED_AWARD", "out", "Award"),
    "prizes":             ("RECEIVED_AWARD", "out", "Award"),
}

# Các mục (section) được phân tích bằng NLP
SECTION_PATTERN = (
    r"(career|research|scientific|work|contribution|controvers|criticism|"
    r"dispute|debate|legacy|later life|academic|influence|reception)"
)

RELATION_TYPES = [
    # từ Infobox
    "DOCTORAL_ADVISOR_OF", "ACADEMIC_ADVISOR_OF", "INFLUENCED",
    "WORKED_AT", "STUDIED_AT", "WORKS_IN_FIELD", "RECEIVED_AWARD",
    # từ NLP trên văn bản
    "BUILT_UPON", "CRITICIZED", "REFUTED", "PRIORITY_DISPUTE",
    "COLLABORATED_WITH", "ADVISOR_TO",
]
