"""Mô hình nhận dạng quan hệ (Relation Extraction) học bằng giám sát xa (distant supervision).

Ý tưởng (Mintz et al., 2009): nếu đồ thị (từ Infobox) đã biết  (Bohr, DOCTORAL_ADVISOR_OF, Kramers)
thì câu nhắc tới cả Bohr và Kramers nhiều khả năng diễn đạt quan hệ đó -> dùng làm nhãn huấn luyện.
Cặp (chủ thể, đối tượng) không có cạnh nào -> nhãn NONE.

Mỗi cặp ứng viên = (chủ thể của trang, một mention đã liên kết trong cùng câu).
Đặc trưng (thưa, DictVectorizer):
  - nhãn của đối tượng (Scientist/Institution/Award/Field)
  - động từ chi phối + giới từ chi phối đối tượng ("study"+"under", "work"+"at", "win")
  - đường đi phụ thuộc rút gọn từ đối tượng lên gốc câu
  - túi lemma giữa chủ thể và đối tượng, cửa sổ 3 lemma bên trái đối tượng
  - vị trí tương đối, khoảng cách
Phân loại: One-vs-Rest Logistic Regression -> mỗi loại quan hệ có ngưỡng riêng,
được chọn trên tập kiểm tra để độ chính xác >= mục tiêu (mặc định 0,90).

Nhãn có hướng: "R>" = chủ thể là nguồn (chủ thể R đối tượng), "R<" = đối tượng là nguồn.
"""
from collections import Counter
from typing import Dict, List, Optional, Tuple

PRONOUNS = {"he", "she", "his", "her", "him"}
STOP = {"the", "a", "an", "of", "and", "in", "on", "to", "be", "for", "with", "by", "as", "at",
        "from", "that", "this", "which", "who", ",", ".", "(", ")", "'s", "-", "–", "his", "her",
        "he", "she", "it", "its", "their", "they"}

# nhãn được mô hình hoá (có hướng) và loại đối tượng tương ứng
LABELS = {
    "DOCTORAL_ADVISOR_OF<": "Scientist",   # đối tượng là thầy hướng dẫn tiến sĩ của chủ thể
    "DOCTORAL_ADVISOR_OF>": "Scientist",   # chủ thể hướng dẫn tiến sĩ cho đối tượng
    "ACADEMIC_ADVISOR_OF<": "Scientist",
    "ACADEMIC_ADVISOR_OF>": "Scientist",
    "INFLUENCED<": "Scientist",
    "INFLUENCED>": "Scientist",
    "COLLABORATED_WITH>": "Scientist",
    "STUDIED_AT>": "Institution",
    "WORKED_AT>": "Institution",
    "RECEIVED_AWARD>": "Award",
    "WORKS_IN_FIELD>": "Field",
}


def subject_token(sent, self_spans: List[Tuple[int, int]], surname: str,
                  other_spans: Optional[List[Tuple[int, int]]] = None):
    """Token đại diện cho chủ thể của trang trong câu (tên/họ/đại từ), hoặc None.
    Họ nằm bên trong tên của một thực thể KHÁC ("Jerome Karle" trên trang Isabella Karle) bị bỏ qua."""
    other_spans = other_spans or []
    for t in sent:
        if any(a <= t.idx < b for a, b in self_spans):
            return t
    for t in sent:
        if t.lower_ == surname and t.pos_ == "PROPN" and not any(a <= t.idx < b for a, b in other_spans):
            return t
    for t in sent:
        if t.lower_ in PRONOUNS and t.dep_ in ("nsubj", "nsubjpass", "poss"):
            return t
    return None


def _governor(tok):
    """Đi lên cây tới động từ / danh từ chi phối đầu tiên, ghi lại giới từ trên đường."""
    prep, cur, steps = None, tok, 0
    while cur.head.i != cur.i and steps < 6:
        cur = cur.head; steps += 1
        if cur.pos_ == "ADP" and prep is None:
            prep = cur.lemma_.lower()
        if cur.pos_ in ("VERB", "AUX") or (cur.pos_ == "NOUN" and cur.dep_ in ("ROOT", "nsubj", "attr", "dobj")):
            return cur, prep
    return cur, prep


def _path(tok, limit=5):
    out, cur = [], tok
    for _ in range(limit):
        out.append(cur.dep_)
        if cur.pos_ in ("VERB", "ADP", "NOUN") and cur is not tok:
            out.append(cur.lemma_.lower())
        if cur.head.i == cur.i:
            break
        cur = cur.head
    return "/".join(out)


def pair_features(doc, sent, subj_tok, obj_span, obj_label: str) -> Dict[str, float]:
    o_head = obj_span.root
    f: Dict[str, float] = {f"olab={obj_label}": 1.0, f"odep={o_head.dep_}": 1.0}
    gov, prep = _governor(o_head)
    f[f"gov={gov.lemma_.lower()}"] = 1.0
    if prep:
        f[f"prep={prep}"] = 1.0
        f[f"gov+prep={gov.lemma_.lower()}+{prep}"] = 1.0
    f[f"path={_path(o_head)}"] = 1.0
    f[f"olab+gov={obj_label}+{gov.lemma_.lower()}"] = 1.0
    left = [t for t in doc[max(sent.start, obj_span.start - 5):obj_span.start] if not t.is_punct]
    for k, t in enumerate(reversed(left)):
        lw = t.lemma_.lower()
        if lw not in STOP:
            f[f"L={lw}"] = 1.0
        if k < 2:
            f[f"L{k}={lw}"] = 1.0                                # vị trí sát đối tượng: "under X", "at X"
    if len(left) >= 2:
        f[f"LB={left[-2].lemma_.lower()}_{left[-1].lemma_.lower()}"] = 1.0
    right = [t for t in doc[obj_span.end:min(sent.end, obj_span.end + 4)] if not t.is_punct]
    for t in right:
        if t.lemma_.lower() not in STOP:
            f[f"R={t.lemma_.lower()}"] = 1.0
    g2 = gov.head
    f[f"gov2={g2.lemma_.lower()}"] = 1.0
    f[f"olab+prep={obj_label}+{prep}"] = 1.0
    if subj_tok is not None:
        a, b = sorted([subj_tok.i, obj_span.start])
        b = min(b, obj_span.start if subj_tok.i < obj_span.start else subj_tok.i)
        between = [t for t in doc[a + 1:b] if t.lemma_.lower() not in STOP and not t.is_punct]
        for t in between[:12]:
            f[f"B={t.lemma_.lower()}"] = 1.0
        f["subj_first" if subj_tok.i < obj_span.start else "obj_first"] = 1.0
        d = abs(obj_span.start - subj_tok.i)
        f[f"dist={'<4' if d < 4 else '<10' if d < 10 else '>=10'}"] = 1.0
        f[f"subjdep={subj_tok.dep_}"] = 1.0
        if subj_tok.head.i == gov.i:
            f["subj_shares_gov"] = 1.0
    for t in sent:
        if t.lemma_.lower() in ("doctorate", "phd", "doctoral", "thesis", "dissertation", "supervise",
                                "supervision", "advisor", "adviser", "mentor", "student", "postdoctoral",
                                "influence", "inspire", "collaborate", "collaboration", "award", "prize",
                                "medal", "win", "receive", "professor", "lecturer", "join", "appoint"):
            f[f"S={t.lemma_.lower()}"] = 1.0
    return f


def edge_labels(page: str, obj: str, out_edges: Dict[Tuple[str, str], set]) -> List[str]:
    labs = []
    for r in out_edges.get((page, obj), ()):
        if f"{r}>" in LABELS:
            labs.append(f"{r}>")
    for r in out_edges.get((obj, page), ()):
        if r == "COLLABORATED_WITH":
            labs.append("COLLABORATED_WITH>")
        elif f"{r}<" in LABELS:
            labs.append(f"{r}<")
    return sorted(set(labs))


def to_edge(page: str, obj: str, label: str) -> Tuple[str, str, str]:
    r, d = label[:-1], label[-1]
    return (page, obj, r) if d == ">" else (obj, page, r)


def _ancestors(t):
    out = [t]
    while t.head.i != t.i and len(out) < 40:      # so sánh chỉ số: spaCy tạo object Token mới mỗi lần truy cập
        t = t.head; out.append(t)
    return out


def syntactic_check(sent, subj_tok, obj_span) -> bool:
    """Bộ lọc độ chính xác cao cho cạnh MỚI.
    Tổ tiên chung gần nhất (LCA) của chủ thể và đối tượng phải là một vị từ không bị phủ định;
    chủ thể cách LCA tối đa 2 bước và đóng vai chủ ngữ / sở hữu / tác nhân (không phải đối tượng
    của một mệnh đề phụ nói về người khác)."""
    if subj_tok is None or subj_tok.lower_ in PRONOUNS:      # cạnh mới: chủ thể phải được gọi tên
        return False
    a_s, a_o = _ancestors(subj_tok), _ancestors(obj_span.root)
    ids_o = {t.i for t in a_o}
    lca = next((t for t in a_s if t.i in ids_o), None)
    if lca is None or lca.pos_ not in ("VERB", "AUX", "NOUN"):
        return False
    if a_s.index(lca) > 2:
        return False
    path_s = a_s[:a_s.index(lca)]
    if path_s and path_s[-1].dep_ not in ("nsubj", "nsubjpass", "poss", "agent", "appos", "compound", "pobj"):
        return False
    if any(c.dep_ == "neg" for c in lca.children) or any(c.dep_ == "neg" for c in a_o[:3] for c in c.children):
        return False
    return True
