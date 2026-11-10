"""Trích xuất quan hệ giữa các nhà khoa học từ văn bản Wikipedia.

Quy trình cho mỗi mục văn bản (section):
  1. Tách câu + tách từ tố + gán nhãn từ loại + phân tích phụ thuộc (spaCy).
  2. Nhận dạng thực thể: dùng wikilink (đã liên kết sẵn) + NER của spaCy
     làm tín hiệu phụ (chỉ giữ wikilink có nhãn NER PERSON/ORG hoặc không nhãn).
  3. Phân giải đồng tham chiếu đơn giản: đại từ ngôi 3 (he/she/his/her) và họ
     (surname) của chủ thể bài viết được ánh xạ về chủ thể bài viết.
  4. Phát hiện "trigger" quan hệ bằng từ điển bổ ngữ theo LEMMA (giống bài toán
     trích xuất sự kiện: trigger + tham số + vai trò).
  5. Xác định hướng quan hệ bằng cây phụ thuộc: chủ ngữ (nsubj) là tác nhân,
     tân ngữ / giới từ là đối tượng; câu bị động ("was criticized by X") đảo hướng.
  6. (Tuỳ chọn) Kiểm chứng bằng mô hình NLI dựa trên Transformer (zero-shot):
     giả thuyết "A criticized B" phải được câu bằng chứng suy ra (entailment).
"""
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set

import spacy

from .wikitext import Section, LinkSpan

# ---------------------------------------------------------------------------
# Từ điển trigger (theo lemma). Mỗi mục: (mẫu regex trên chuỗi lemma, hướng)
#   hướng "subj->obj": chủ ngữ thực hiện hành động lên tân ngữ
#   "sym": quan hệ đối xứng
# ---------------------------------------------------------------------------
TRIGGERS = {
    "BUILT_UPON": [
        r"\bbuild (on|upon)\b", r"\bbase on (the )?(work|idea|theory|result|finding)s? of\b",
        r"\bextend\b", r"\bgeneralize\b", r"\bgeneralise\b", r"\belaborate (on|upon)\b",
        r"\bdraw (on|upon)\b", r"\binspire by\b", r"\bfollow(ing)? (the )?(work|idea|lead)s? of\b",
        r"\bdevelop .{0,30}'s (theory|idea|work|model|method)\b", r"\brefine\b",
        r"\bbuild on the work\b", r"\bexpand (on|upon)\b",
    ],
    "CRITICIZED": [
        r"\bcriticize\b", r"\bcriticise\b", r"\bcritique\b", r"\bbe critical of\b",
        r"\bcriticism of\b", r"\bchallenge\b", r"\bobject to\b", r"\battack\b",
        r"\bbe skeptical of\b", r"\bbe sceptical of\b", r"\bquestion\b", r"\boppose\b",
        r"\bclash with\b", r"\bdenounce\b",
    ],
    "REFUTED": [
        r"\brefute\b", r"\bdisprove\b", r"\brebut\b", r"\breject\b", r"\bdebunk\b",
        r"\bcontradict\b", r"\boverturn\b", r"\bfalsify\b", r"\bdiscredit\b",
        r"\bshow .{0,40} (be )?(wrong|incorrect|false|flawed)\b",
    ],
    "PRIORITY_DISPUTE": [
        r"\bpriority dispute\b", r"\bdispute over priority\b", r"\bclaim priority\b",
        r"\bpriority (claim|controversy)\b", r"\baccuse .{0,40}plagiari[sz]\b",
        r"\bplagiari[sz]e\b", r"\bplagiarism\b", r"\bclaim credit\b", r"\bdispute .{0,30}credit\b",
    ],
    "COLLABORATED_WITH": [
        r"\bcollaborate with\b", r"\bin collaboration with\b", r"\bcollaboration with\b",
        r"\bwork (together )?with\b", r"\btogether with\b", r"\bjointly with\b",
        r"\bco-?author\b", r"\bco-?discover\b", r"\bco-?found\b", r"\bpartner with\b",
        r"\balong with\b", r"\bwith his (colleague|collaborator|student)\b",
        r"\bwith her (colleague|collaborator|student)\b",
    ],
    "ADVISOR_TO": [
        r"\badvis[eo]r (to|for|of|at)\b", r"\bconsultant (to|for|at)\b",
        r"\bserve on the (scientific )?advisory\b", r"\b(scientific )?advisory (board|committee|council)\b",
        r"\badvise\b", r"\bconsult for\b", r"\bscientific council\b",
    ],
}
SYMMETRIC = {"COLLABORATED_WITH", "PRIORITY_DISPUTE"}
NEGATION = re.compile(r"\b(not|never|no|neither|nor|without)\b")
# Thứ tự ưu tiên khi một câu khớp nhiều trigger
PRIORITY = ["PRIORITY_DISPUTE", "REFUTED", "CRITICIZED", "BUILT_UPON", "ADVISOR_TO", "COLLABORATED_WITH"]
_COMPILED = {k: [re.compile(p) for p in v] for k, v in TRIGGERS.items()}
PRONOUNS = {"he", "she", "him", "her", "his", "hers"}


@dataclass
class Relation:
    source: str
    target: str
    rel_type: str
    evidence: str
    section: str
    page: str
    trigger: str
    method: str = "rule+dep"
    confidence: float = 0.0


class RelationExtractor:
    def __init__(self, model: str = "en_core_web_sm", nli_model: Optional[str] = None,
                 nli_threshold: float = 0.7):
        self.nlp = spacy.load(model)
        self.nli = None
        self.nli_threshold = nli_threshold
        if nli_model:
            from transformers import pipeline  # import trễ: tuỳ chọn
            self.nli = pipeline("zero-shot-classification", model=nli_model)

    # ------------------------------------------------------------------
    def extract(self, page_title: str, sections: List[Section], node_types: Dict[str, str]) -> List[Relation]:
        """node_types: {title: Scientist|Institution|...} cho các trang được link tới."""
        surname = page_title.split("(")[0].strip().split()[-1].lower() if page_title else ""
        rels: List[Relation] = []
        for sec in sections:
            doc = self.nlp(sec.text)
            for sent in doc.sents:
                rels += self._from_sentence(page_title, surname, sec, sent, node_types)
        return self._dedupe(rels)

    # ------------------------------------------------------------------
    def _from_sentence(self, page, surname, sec: Section, sent, node_types) -> List[Relation]:
        s0, s1 = sent.start_char, sent.end_char
        text = sent.text.strip()
        if len(text) < 20 or len(text) > 600:
            return []
        lemma_str = " ".join(t.lemma_.lower() for t in sent)
        lemma_str = re.sub(r"\s+'s\b", "'s", lemma_str)

        # -- 1. Ứng viên thực thể: wikilink trong câu, đã biết loại nút ----------
        mentions = []
        for l in sec.links:
            if l.start >= s0 and l.end <= s1 and l.title != page and l.title in node_types:
                mentions.append(l)
        if not mentions:
            return []

        # -- 2. Đồng tham chiếu: vị trí nhắc tới chủ thể bài viết ---------------
        subj_tokens = [t for t in sent if t.lower_ in PRONOUNS or
                       (surname and t.lower_ == surname and not any(m.start <= t.idx < m.end for m in mentions))]
        subject_present = bool(subj_tokens) or sec.heading == "Lead"

        # -- 3. Trigger --------------------------------------------------------
        found = []
        for rtype in PRIORITY:
            for rx in _COMPILED[rtype]:
                m = rx.search(lemma_str)
                if m:
                    found.append((rtype, m.group(0)))
                    break
        if not found:
            return []
        if NEGATION.search(lemma_str):
            # câu phủ định ("did not reject") -> bỏ để giữ độ chính xác cao
            return []

        out = []
        for rtype, trig in found[:1]:          # chỉ nhận trigger ưu tiên cao nhất
            for m in mentions:
                ttype = node_types.get(m.title)
                if rtype == "ADVISOR_TO":
                    if ttype != "Institution":
                        continue
                elif ttype != "Scientist":
                    continue
                if not subject_present and not self._subject_is_agent(sent):
                    continue
                direction = self._direction(sent, m, rtype, trig)
                if direction is None:
                    continue
                src, dst = (page, m.title) if direction == "out" else (m.title, page)
                if rtype in SYMMETRIC:
                    src, dst = sorted([page, m.title])
                conf = self._score(sent, m, trig, subject_present)
                out.append(Relation(src, dst, rtype, text, sec.heading, page, trig, confidence=conf))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _subject_is_agent(sent) -> bool:
        return any(t.dep_ in ("nsubj", "nsubjpass") and t.lower_ in PRONOUNS for t in sent)

    @staticmethod
    def _direction(sent, m: LinkSpan, rtype: str, trig: str) -> Optional[str]:
        """'out' = chủ thể bài viết -> thực thể m ; 'in' = m -> chủ thể."""
        if rtype in SYMMETRIC:
            return "out"
        toks = [t for t in sent if m.start <= t.idx < m.end]
        if not toks:
            return None
        head = max(toks, key=lambda t: len(list(t.subtree)))
        deps = {head.dep_} | {a.dep_ for a in head.ancestors}
        passive = any(t.dep_ in ("nsubjpass", "auxpass") for t in sent)
        # Thực thể m nằm trong cụm "by X" của câu bị động -> m là tác nhân
        if any(a.dep_ == "agent" for a in head.ancestors) or head.dep_ == "agent":
            return "in"
        # m là chủ ngữ chủ động -> m là tác nhân
        if head.dep_ == "nsubj" or ("nsubj" in deps and "pobj" not in deps and "dobj" not in deps):
            return "in"
        # m là chủ ngữ bị động: "X's theory was refuted by him" -> chủ thể là tác nhân
        if head.dep_ == "nsubjpass" or "nsubjpass" in deps:
            return "out"
        # m là tân ngữ / bổ ngữ giới từ / sở hữu cách ("criticized X's theory")
        if deps & {"dobj", "pobj", "poss", "attr", "dative", "conj", "appos", "compound", "npadvmod"}:
            return "in" if (passive and rtype == "BUILT_UPON" and "pobj" not in deps) else "out"
        return None

    @staticmethod
    def _score(sent, m: LinkSpan, trig: str, subject_present: bool) -> float:
        score = 0.6
        if subject_present:
            score += 0.15
        if len(sent) <= 40:
            score += 0.1
        # khoảng cách trigger - thực thể (theo ký tự)
        idx = sent.text.lower().find(trig.split()[0])
        if idx >= 0 and abs((m.start - sent.start_char) - idx) < 80:
            score += 0.1
        return round(min(score, 0.95), 3)

    # ------------------------------------------------------------------
    def verify_with_nli(self, rels: List[Relation]) -> List[Relation]:
        """Kiểm chứng bằng NLI (BERT/BART MNLI). Giữ lại quan hệ có entailment cao."""
        if not self.nli:
            return rels
        verb = {"BUILT_UPON": "built upon the work of", "CRITICIZED": "criticized",
                "REFUTED": "refuted the work of", "PRIORITY_DISPUTE": "had a priority dispute with",
                "COLLABORATED_WITH": "collaborated with", "ADVISOR_TO": "was an advisor to"}
        kept = []
        for r in rels:
            hyp = f"{r.source} {verb[r.rel_type]} {r.target}."
            res = self.nli(r.evidence, candidate_labels=[hyp], hypothesis_template="{}", multi_label=True)
            p = float(res["scores"][0])
            if p >= self.nli_threshold:
                r.method = "rule+dep+nli"
                r.confidence = round((r.confidence + p) / 2, 3)
                kept.append(r)
        return kept

    @staticmethod
    def _dedupe(rels: Iterable[Relation]) -> List[Relation]:
        best: Dict[tuple, Relation] = {}
        for r in rels:
            if r.source == r.target:
                continue
            k = (r.source, r.target, r.rel_type)
            if k not in best or r.confidence > best[k].confidence:
                best[k] = r
        return list(best.values())


def to_dicts(rels: List[Relation]) -> List[dict]:
    return [asdict(r) for r in rels]


# =============================================================================
# Phiên bản 2: xác minh vai trò ngữ nghĩa bằng cây phụ thuộc (dùng mặc định)
# Một cạnh chỉ được tạo khi chủ thể bài viết và thực thể là đúng hai "tham số"
# của chính từ trigger (tác nhân / đối tượng), giống trích xuất sự kiện.
# =============================================================================
VERB_TRIGGERS = {
    "CRITICIZED": {"criticize", "criticise", "critique", "challenge", "attack", "oppose", "denounce", "clash", "object"},
    "REFUTED": {"refute", "disprove", "rebut", "reject", "debunk", "contradict", "overturn", "falsify", "discredit"},
    "BUILT_UPON": {"build", "extend", "generalize", "generalise", "elaborate", "draw", "inspire", "refine", "expand", "base"},
    "COLLABORATED_WITH": {"collaborate", "work", "partner", "co-author", "coauthor", "co-discover", "co-found", "cofound", "team"},
    "PRIORITY_DISPUTE": {"plagiarize", "plagiarise"},
}
NEEDS_PREP = {"build": {"on", "upon"}, "draw": {"on", "upon"}, "elaborate": {"on", "upon"}, "expand": {"on", "upon"},
              "base": {"on"}, "work": {"with"}, "team": {"with"}, "partner": {"with"}, "clash": {"with"}, "object": {"to"}}
ADVISOR_NOUNS = {"advisor", "adviser", "consultant", "board", "committee", "council"}
REL_PRON = {"who", "that", "which"}
JOINT_WORK_VERBS = {"design", "discover", "co-discover", "develop", "prove", "found", "co-found", "establish", "conduct",
                    "build", "create", "invent", "study", "investigate", "write", "co-write", "publish", "perform",
                    "propose", "introduce", "derive", "demonstrate", "solve", "construct", "devise", "formulate",
                    "synthesize", "synthesise", "isolate", "identify", "measure", "observe", "calculate", "compute",
                    "work", "collaborate", "research", "experiment", "edit", "author", "co-author", "launch", "organize"}
AWARD_VERBS = {"award", "receive", "share", "win", "elect", "name", "honor", "honour", "give"}
NOUN_TRIGGERS = {"criticism": "CRITICIZED", "critique": "CRITICIZED", "critical": "CRITICIZED", "skeptical": "CRITICIZED",
                 "sceptical": "CRITICIZED", "attack": "CRITICIZED", "opponent": "CRITICIZED", "critic": "CRITICIZED",
                 "refutation": "REFUTED", "rejection": "REFUTED", "disproof": "REFUTED", "rebuttal": "REFUTED"}
PRIORITY_CUE = re.compile(r"priority|credit|plagiar|who (first )?discovered|claim(ed)? to have")


class RoleExtractor(RelationExtractor):
    """Trích xuất dựa trên vai trò của từ trigger trong cây phụ thuộc."""

    def extract(self, page_title, sections, node_types):
        base = page_title.split("(")[0].strip()
        self._subj_words = {w.lower() for w in base.split() if len(w) > 2}
        self._surname = base.split()[-1].lower() if base else ""
        rels = []
        for sec in sections:
            doc = self.nlp(sec.text)
            for sent in doc.sents:
                try:
                    rels += self._roles(page_title, sec, sent, node_types)
                except Exception:
                    continue
        return self._dedupe(rels)

    # -- quy chiếu một token về: ("ENT", title) | ("SUBJ", page) | None
    def _ref(self, tok, spans, depth=0):
        if tok is None or depth > 3:
            return None
        for l in spans:
            if l.start <= tok.idx < l.end:
                return ("ENT", l.title)
        for c in tok.children:          # tên nhiều từ: "physicist James Chadwick"
            if c.dep_ in ("compound", "flat", "appos"):
                for l in spans:
                    if l.start <= c.idx < l.end:
                        return ("ENT", l.title)
        low = tok.lower_
        if low in PRONOUNS:
            return ("SUBJ", None)
        if low == self._surname and tok.pos_ == "PROPN":
            return ("SUBJ", None)
        # "his theory", "Rival's interpretation", "the work of X"
        for c in tok.children:
            if c.dep_ == "poss":
                r = self._ref(c, spans, depth + 1)
                if r: return r
        for c in tok.children:
            if c.dep_ == "prep" and c.lower_ == "of":
                for g in c.children:
                    if g.dep_ == "pobj":
                        r = self._ref(g, spans, depth + 1)
                        if r: return r
        return None

    def _with_conj(self, toks):
        out = []
        for t in toks:
            out.append(t)
            out += [c for c in t.conjuncts]
        return out

    def _agents(self, v):
        ag = [c for c in v.children if c.dep_ in ("nsubj",)]
        ag = [v.head if (a.lower_ in REL_PRON and v.dep_ == "relcl") else a for a in ag]
        if not ag and v.dep_ in ("conj", "xcomp", "advcl") and v.head.pos_ in ("VERB", "AUX"):
            ag = [c for c in v.head.children if c.dep_ == "nsubj"]
        if not ag and v.dep_ == "relcl":
            ag = [v.head]
        by = [g for c in v.children if c.dep_ == "agent" for g in c.children if g.dep_ == "pobj"]
        return self._with_conj(ag), self._with_conj(by)

    def _preps(self, v, words):
        return [g for c in v.children if c.dep_ in ("prep", "prt") and c.lower_ in words
                for g in c.children if g.dep_ == "pobj"]

    def _mk(self, rtype, src, dst, page, sent, sec, trig, node_types):
        if src is None or dst is None or src == dst:
            return None
        if src[0] == "SUBJ": src = ("SUBJ", page)
        if dst[0] == "SUBJ": dst = ("SUBJ", page)
        kinds = {src[0], dst[0]}
        if kinds != {"SUBJ", "ENT"}:
            return None
        ent = src[1] if src[0] == "ENT" else dst[1]
        need = "Institution" if rtype == "ADVISOR_TO" else "Scientist"
        if node_types.get(ent) != need:
            return None
        a, b = src[1], dst[1]
        if rtype in SYMMETRIC:
            a, b = sorted([a, b])
        conf = 0.85 if len(sent) <= 40 else 0.75
        return Relation(a, b, rtype, sent.text.strip(), sec.heading, page, trig, method="rule+role", confidence=conf)

    def _roles(self, page, sec, sent, node_types):
        if not (20 <= len(sent.text) <= 600):
            return []
        s0, s1 = sent.start_char, sent.end_char
        spans = [l for l in sec.links if l.start >= s0 and l.end <= s1 and l.title != page and l.title in node_types]
        if not spans:
            return []
        out = []
        for v in sent:
            lem = v.lemma_.lower()
            if any(c.dep_ == "neg" for c in v.children):
                continue
            if any(l.start <= v.idx < l.end for l in spans):   # trigger nằm trong tên riêng (vd "Dimensional Fund Advisors")
                continue
            # ---- động từ trigger
            for rtype, lems in VERB_TRIGGERS.items():
                if lem not in lems or v.pos_ not in ("VERB", "AUX", "NOUN") and lem not in ("co-author",):
                    continue
                if v.pos_ == "NOUN" and rtype != "COLLABORATED_WITH":
                    continue
                need = NEEDS_PREP.get(lem)
                passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in v.children)
                agents, by = self._agents(v)
                patients = [c for c in v.children if c.dep_ in ("dobj", "dative", "attr")]
                if need:
                    patients = self._preps(v, need)
                    if not patients and lem not in ("work",):
                        continue
                patients = self._with_conj(patients)
                subjpass = self._with_conj([c for c in v.children if c.dep_ == "nsubjpass"])
                pairs = []
                if rtype == "COLLABORATED_WITH":
                    withs = self._with_conj(self._preps(v, {"with"}) + patients)
                    pairs = [(a, w) for a in agents for w in withs]
                elif lem in ("inspire",) and passive:          # B was inspired by A -> B built upon A
                    pairs = [(p_, a) for p_ in subjpass for a in by]
                elif lem in ("inspire",):                       # A('s work) inspired B -> B built upon A
                    pairs = [(p_, a) for a in agents for p_ in patients]
                elif lem == "base" and passive:                 # his work was based on X
                    pairs = [(p_, x) for p_ in subjpass for x in self._with_conj(self._preps(v, {"on", "upon"}))]
                elif passive:                                   # X's theory was refuted by Y -> Y -> X
                    pairs = [(a, p_) for a in by for p_ in subjpass]
                else:
                    pairs = [(a, p_) for a in agents for p_ in patients]
                for a, b in pairs:
                    r = self._mk(rtype, self._ref(a, spans), self._ref(b, spans), page, sent, sec, lem, node_types)
                    if r: out.append(r)
            # ---- "together with X", "along with X", "in collaboration with X"
            if lem == "with" and v.dep_ == "prep" and v.head.lower_ in ("together", "along", "jointly", "collaboration"):
                gov = v.head
                while gov.dep_ not in ("ROOT",) and gov.pos_ not in ("VERB", "AUX"):
                    gov = gov.head
                if gov.lemma_.lower() in AWARD_VERBS or any(t.lemma_.lower() in ("prize", "award", "medal") for t in gov.subtree):
                    continue
                # "along with X, he was regarded as..." không phải hợp tác: chỉ nhận khi động từ là hành động nghiên cứu chung
                if v.head.lower_ in ("along", "together") and gov.lemma_.lower() not in JOINT_WORK_VERBS:
                    continue
                agents = self._with_conj([c for c in gov.children if c.dep_ in ("nsubj", "nsubjpass")])
                for w in self._with_conj([c for c in v.children if c.dep_ == "pobj"]):
                    for a in agents:
                        r = self._mk("COLLABORATED_WITH", self._ref(a, spans), self._ref(w, spans), page, sent, sec, "together with", node_types)
                        if r: out.append(r)
            # ---- cố vấn: "advisor to X", "served on the advisory board of X", "consultant for X"
            if lem in ADVISOR_NOUNS and v.pos_ == "NOUN":
                if lem in ("board", "committee", "council") and not any(c.lower_ in ("advisory", "scientific") for c in v.children):
                    continue
                orgs = self._preps(v, {"to", "for", "of", "at"})
                orgs += [c for c in v.children if c.dep_ in ("compound",)]
                gov = v
                while gov.pos_ not in ("VERB", "AUX") and gov.dep_ != "ROOT":
                    gov = gov.head
                holders = self._with_conj([c for c in gov.children if c.dep_ in ("nsubj", "nsubjpass")])
                for o in orgs:
                    for h in holders:
                        r = self._mk("ADVISOR_TO", self._ref(h, spans), self._ref(o, spans), page, sent, sec, lem, node_types)
                        if r: out.append(r)
            if lem in ("advise", "consult") and v.pos_ == "VERB":
                agents, _ = self._agents(v)
                objs = [c for c in v.children if c.dep_ == "dobj"] + self._preps(v, {"for"})
                for a in agents:
                    for o in objs:
                        r = self._mk("ADVISOR_TO", self._ref(a, spans), self._ref(o, spans), page, sent, sec, lem, node_types)
                        if r: out.append(r)
            # ---- danh từ / tính từ: "his criticism of X", "a refutation of X's theory", "was critical of X"
            if lem in NOUN_TRIGGERS and v.pos_ in ("NOUN", "ADJ"):
                rtype = NOUN_TRIGGERS[lem]
                targets = self._with_conj(self._preps(v, {"of", "against"}))
                holders = [c for c in v.children if c.dep_ == "poss"]
                gov = v
                while gov.pos_ not in ("VERB", "AUX") and gov.dep_ != "ROOT":
                    gov = gov.head
                if v.pos_ == "ADJ" or not holders:
                    holders += [c for c in gov.children if c.dep_ in ("nsubj",)]
                for h in self._with_conj(holders):
                    for t_ in targets:
                        r = self._mk(rtype, self._ref(h, spans), self._ref(t_, spans), page, sent, sec, lem, node_types)
                        if r: out.append(r)
            # ---- "dispute/controversy with X over priority / credit"
            if lem in ("dispute", "controversy", "feud", "quarrel") and v.pos_ == "NOUN" and PRIORITY_CUE.search(sent.text.lower()):
                gov = v
                while gov.pos_ not in ("VERB", "AUX") and gov.dep_ != "ROOT":
                    gov = gov.head
                others = self._with_conj(self._preps(v, {"with", "between"}) + self._preps(gov, {"with", "between"}))
                holders = self._with_conj([c for c in gov.children if c.dep_ in ("nsubj", "nsubjpass")] + [c for c in v.children if c.dep_ == "poss"]) + others
                for a in holders:
                    for b in others:
                        r = self._mk("PRIORITY_DISPUTE", self._ref(a, spans), self._ref(b, spans), page, sent, sec, "dispute (priority)", node_types)
                        if r: out.append(r)
            # ---- tranh chấp ưu tiên / đạo văn
            if lem in ("dispute", "controversy") and any(c.lower_ == "priority" for c in v.children):
                gov = v
                while gov.pos_ not in ("VERB", "AUX") and gov.dep_ != "ROOT":
                    gov = gov.head
                others = self._with_conj(self._preps(v, {"with", "between"}) + self._preps(gov, {"with", "between"}))
                holders = self._with_conj([c for c in gov.children if c.dep_ in ("nsubj", "nsubjpass")]) + others
                for a in holders:
                    for b in others:
                        r = self._mk("PRIORITY_DISPUTE", self._ref(a, spans), self._ref(b, spans), page, sent, sec, "priority dispute", node_types)
                        if r: out.append(r)
            if lem == "accuse" and v.pos_ == "VERB":
                ofs = self._preps(v, {"of"})
                if any(o.lemma_.lower().startswith("plagiar") or o.lemma_.lower() in ("steal", "theft") for o in ofs):
                    agents, by = self._agents(v)
                    objs = [c for c in v.children if c.dep_ in ("dobj", "nsubjpass")]
                    for a in agents + by:
                        for o in objs:
                            r = self._mk("PRIORITY_DISPUTE", self._ref(a, spans), self._ref(o, spans), page, sent, sec, "accuse of plagiarism", node_types)
                            if r: out.append(r)
        return out
