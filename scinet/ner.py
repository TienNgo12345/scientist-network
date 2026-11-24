"""Mô hình nhận dạng thực thể (NER) + liên kết thực thể (Entity Linking) cho miền nhà khoa học.

Ba tầng:
  1. Phát hiện mention (mention detection)
     a. NER thống kê của spaCy (PERSON, ORG, FAC, WORK_OF_ART, NORP bị loại)
     b. Gazetteer (PhraseMatcher, phân biệt hoa thường) xây từ tên + bí danh của 9.303 nút
        trong đồ thị và anchor text của 41 nghìn wikilink -> bắt được Award / Field / Institution
        mà NER thống kê hay bỏ sót.
     Hai nguồn được gộp, ưu tiên span dài hơn.
  2. Sinh ứng viên (candidate generation) từ bảng bí danh: surface -> {title: số lần xuất hiện}.
  3. Xếp hạng ứng viên:  score = P(title | surface)  (prior từ anchor text)
                               + 0.5 nếu title đã xuất hiện trong ngữ cảnh trang (page context)
     Tên một từ (họ) chỉ được liên kết khi trong trang có đúng một người mang họ đó.

Mỗi mention trả về: start, end, text, label (Scientist/Institution/Award/Field/PERSON/ORG),
kb (title trong đồ thị hoặc None = thực thể mới), score.
"""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from spacy.matcher import PhraseMatcher

SPACY_KEEP = {"PERSON": "PERSON", "ORG": "ORG", "FAC": "ORG", "WORK_OF_ART": "AWARD?"}
STOP_SURFACES = {"he", "she", "his", "her", "the", "university", "institute", "college",
                 "school", "prize", "medal", "award", "society", "academy", "science",
                 "professor", "laboratory", "department", "faculty", "royal", "national",
                 "fellow", "member", "president", "director", "doctor", "nobel"}
_STRIP_RE = re.compile(r"^(the|a|an)\s+|['’]s$|\s*\(.*?\)\s*$", re.I)


@dataclass
class Mention:
    start: int
    end: int
    text: str
    label: str
    kb: Optional[str]
    score: float
    source: str  # "ner" | "gazetteer" | "both"


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().strip(".,;:()\"'"))
    for _ in range(2):
        s = _STRIP_RE.sub("", s).strip()
    return s.lower()


AWARD_KW = re.compile(r"prize|medal|award|lecture|order|cross|laureate|fellow|trophy|cup|"
                      r"prix|preis|premio|medaille|ratna|bhushan|shri|vibhushan|imtiaz|knight bachelor|knighthood|^frse?$|"
                      r"scholarship|membership|grant|honou?r|chair", re.I)


def linkable(title: str, label: str) -> bool:
    """Nút Award một từ không có từ khoá giải thưởng (vd 'Cambridge', 'Knight', 'Sciences')
    thường là rác do Infobox ghi chữ thường -> không cho NER liên kết tới."""
    if label != "Award":
        return True
    return bool(AWARD_KW.search(title)) or (len(title.split()) >= 2 and title.istitle())


class AliasTable:
    """surface (lower) -> Counter(title) ; title -> nhãn trong đồ thị."""

    def __init__(self, node_labels: Dict[str, str]):
        node_labels = {t: l for t, l in node_labels.items() if linkable(t, l)}
        self.node_labels = node_labels
        self.table: Dict[str, Counter] = defaultdict(Counter)
        for t in node_labels:
            self.add(t, t, 3)
            base = re.sub(r"\s*\(.*?\)\s*$", "", t)          # "Max Born (physicist)" -> "Max Born"
            if base != t:
                self.add(base, t, 3)

    def add(self, surface: str, title: str, w: int = 1):
        if title not in self.node_labels:
            return
        s = _norm(surface)
        if len(s) < 3 or s in STOP_SURFACES:
            return
        self.table[s][title] += w

    def add_redirects(self, aliases: Dict[str, str]):
        for a, t in aliases.items():
            self.add(a, t, 2)

    def candidates(self, surface: str) -> Counter:
        return self.table.get(_norm(surface), Counter())


class EntityRecognizer:
    def __init__(self, nlp, node_labels: Dict[str, str], alias: AliasTable,
                 use_gazetteer: bool = True, use_spacy: bool = True, use_context: bool = True):
        self.nlp, self.labels, self.alias = nlp, node_labels, alias
        self.use_gaz, self.use_spacy, self.use_ctx = use_gazetteer, use_spacy, use_context
        self.matcher = PhraseMatcher(nlp.vocab, attr="ORTH")      # phân biệt hoa thường
        pats = []
        for s, cnt in alias.table.items():
            title = cnt.most_common(1)[0][0]
            # chỉ đưa vào gazetteer các bề mặt viết hoa (tên riêng) xuất hiện nguyên dạng
            for surf in {title, re.sub(r"\s*\(.*?\)\s*$", "", title)}:
                if _norm(surf) == s and surf[:1].isupper() and len(surf) >= 4:
                    pats.append(surf)
        self._gaz_surfaces = sorted(set(pats))
        for i in range(0, len(self._gaz_surfaces), 5000):
            self.matcher.add("KB", list(nlp.tokenizer.pipe(self._gaz_surfaces[i:i + 5000])))
        # Lĩnh vực thường viết thường trong câu ("quantum mechanics") -> matcher không phân biệt hoa thường
        self.field_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        fields = [s for s, cnt in alias.table.items()
                  if node_labels.get(cnt.most_common(1)[0][0]) == "Field" and len(s) >= 5]
        if fields:
            self.field_matcher.add("FIELD", list(nlp.tokenizer.pipe(fields)))

    def add_surface_patterns(self, surfaces: Iterable[str]):
        extra = [s for s in set(surfaces) if s[:1].isupper() and len(s) >= 4]
        if extra:
            self.matcher.add("KB", list(self.nlp.tokenizer.pipe(extra)))

    # ------------------------------------------------------------------
    def recognize(self, doc, page_title: Optional[str] = None,
                  page_context: Optional[Set[str]] = None) -> List[Mention]:
        ctx = page_context if (self.use_ctx and page_context) else set()
        surname_map = self._surname_map(ctx | ({page_title} if page_title else set()))
        gaz = []
        if self.use_gaz:
            for _, a, b in list(self.matcher(doc)) + list(self.field_matcher(doc)):
                sp = doc[a:b]
                kb, sc = self._link(sp.text, None, ctx, surname_map, page_title)
                if kb:
                    gaz.append(Mention(sp.start_char, sp.end_char, sp.text, self.labels[kb], kb, sc,
                                       "gazetteer"))
        stat = []
        if self.use_spacy:
            for e in doc.ents:
                if e.label_ not in SPACY_KEEP:
                    continue
                lab = SPACY_KEEP[e.label_]
                kb, sc = self._link(e.text, lab, ctx, surname_map, page_title)
                inner = [g for g in gaz if g.start >= e.start_char and g.end <= e.end_char]
                if kb:
                    src = "both" if any(g.kb == kb for g in inner) else "ner"
                    stat.append(Mention(e.start_char, e.end_char, e.text, self.labels[kb], kb, sc, src))
                elif not inner:
                    stat.append(Mention(e.start_char, e.end_char, e.text,
                                        "ORG" if lab == "AWARD?" else lab, None, 0.0, "ner"))
        # Gộp: ưu tiên mention đã liên kết, rồi span dài hơn
        allm = sorted(stat + gaz, key=lambda m: (m.kb is None, -(m.end - m.start), m.start))
        keep = []
        for m in allm:
            if any(not (m.end <= k.start or m.start >= k.end) for k in keep):
                continue
            keep.append(m)
        return sorted(keep, key=lambda m: m.start)

    @staticmethod
    def _resolve_overlaps(spans):
        spans = sorted(spans, key=lambda x: (-(x[1] - x[0]), x[0]))
        taken, keep = [], []
        for s, e, lab, src in spans:
            if any(not (e <= a or s >= b) for a, b in taken):
                # cùng đúng span ở cả 2 nguồn -> đánh dấu both
                for i, (a, b, l2, s2) in enumerate(keep):
                    if (a, b) == (s, e) and s2 != src:
                        keep[i] = (a, b, l2 or lab, "both")
                continue
            taken.append((s, e)); keep.append((s, e, lab, src))
        return sorted(keep)

    def _surname_map(self, titles: Set[str]) -> Dict[str, str]:
        cnt = defaultdict(set)
        for t in titles:
            if self.labels.get(t) == "Scientist":
                base = re.sub(r"\s*\(.*?\)\s*$", "", t)
                cnt[base.split()[-1].lower()].add(t)
        return {k: next(iter(v)) for k, v in cnt.items() if len(v) == 1}

    def _link(self, text, lab, ctx, surname_map, page_title):
        cands = self.alias.candidates(text)
        best, best_sc = None, 0.0
        if cands:
            total = sum(cands.values())
            for t, c in cands.items():
                sc = c / total + (0.5 if t in ctx else 0.0)
                if sc > best_sc:
                    best, best_sc = t, sc
        toks = text.split()
        if best is None and len(toks) == 1 and lab in ("PERSON", None):
            t = surname_map.get(toks[0].lower().strip(".,'s’"))
            if t:
                best, best_sc = t, 0.6
        if best is None and len(toks) >= 2 and lab == "PERSON":
            # "Niels Henrik David Bohr" -> khớp họ + tên đầu với một người trong ngữ cảnh
            last = toks[-1].lower()
            t = surname_map.get(last)
            if t and t.split()[0].lower() == toks[0].lower():
                best, best_sc = t, 0.55
        if best is not None and best == page_title:
            return best, best_sc
        return (best, round(best_sc, 3)) if best_sc >= 0.3 else (None, 0.0)
