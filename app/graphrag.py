"""GraphRAG suy luận nhiều bước (multi-hop) trên đồ thị tri thức nhà khoa học.

Luồng xử lý một câu hỏi:
  1. Liên kết thực thể (KG.link)                       -> nút xuất phát
  2. Lập kế hoạch truy vấn (LLM sinh JSON, kiểm tra hợp lệ; lỗi -> bộ lập kế hoạch theo luật)
        intent ∈ {traverse, path, common, profile, global}, steps = [{rel, dir}], filter, aggregate
  3. Thực thi kế hoạch trên đồ thị -> tập đáp án + CHUỖI SUY LUẬN (các cạnh đã đi qua, có bằng chứng)
  4. Truy hồi ngữ cảnh: dữ kiện 1-hop quanh thực thể (local search), tóm tắt cộng đồng (global search),
     câu bằng chứng gốc trên Wikipedia (BM25)
  5. Sinh câu trả lời bằng LLM chỉ từ ngữ cảnh đã truy hồi (hoặc mẫu câu nếu không có LLM)
"""
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from .kg import REL_VI, Hop, KnowledgeGraph, strip_accents
from .llm import LLM

RELS = list(REL_VI)

# ------------------------------------------------------------------ từ khoá -> quan hệ (luật)
# mỗi mục: (regex trên câu hỏi đã bỏ dấu, quan hệ, hướng nhìn TỪ thực thể đã biết)
KEYWORDS = [
    (r"thay huong dan cua thay|thay cua thay", None, None),  # xử lý bằng tách từng cụm "thay"
    (r"thay huong dan|nguoi huong dan|thay day|\bthay\b|advisor|adviser|supervisor|mentor|teacher", "ADVISOR_UP", None),
    (r"hoc tro|nghien cuu sinh|\btro\b|students?|doctoral students?|proteges?|hoc sinh", "ADVISOR_DOWN", None),
    (r"lam viec|cong tac|giang day|worked|works at|employ|lam o|lam tai", "WORKED_AT", "out"),
    (r"hoc tai|hoc o|tot nghiep|studied|alma mater|educated|theo hoc", "STUDIED_AT", "out"),
    (r"giai thuong|giai|huy chuong|awards?|prizes?|medals?|honou?rs?|doat|nhan duoc", "RECEIVED_AWARD", "out"),
    (r"linh vuc|nganh|fields?|discipline|nghien cuu ve", "WORKS_IN_FIELD", "out"),
    (r"chiu anh huong|influenced by|anh huong boi", "INFLUENCED", "in"),
    (r"anh huong|influenced|influence", "INFLUENCED", "out"),
    (r"hop tac|cong tac voi|collaborat|cung lam viec voi|worked with", "COLLABORATED_WITH", "both"),
    (r"bi phe binh|bi chi trich|criticized by|critics? of", "CRITICIZED", "in"),
    (r"phe binh|chi trich|criticiz|critic", "CRITICIZED", "out"),
    (r"bac bo|refut|disprov", "REFUTED", "out"),
    (r"ke thua|dua tren|phat trien tu|built (up)?on|extended", "BUILT_UPON", "out"),
    (r"tranh chap|priority", "PRIORITY_DISPUTE", "both"),
    (r"co van cho|advis(or|ed) (to|for)|consult", "ADVISOR_TO", "out"),
]
PATH_RE = r"lien he|lien quan|ket noi|duong di|con duong|bac cau|connected|connection|path|relate|how .* linked"
COMMON_RE = r"\bchung\b|\bcung\b(?! lam viec voi)|ca hai|both|in common|shared"
COUNT_RE = r"bao nhieu|so luong|how many|dem"
TOP_RE = r"nhieu nhat|nhat|most|top|xep hang|hang dau"
GLOBAL_RE = r"cong dong|nhom nghien cuu|cum|community|communities|xu huong|trung tam|hat nhan|core|toan mang|ca mang"
WHO_RE = r"\bai\b|who|nhung nguoi|nguoi nao|nha khoa hoc nao"


@dataclass
class Plan:
    intent: str = "traverse"
    entities: List[str] = field(default_factory=list)
    steps: List[dict] = field(default_factory=list)
    filter: Optional[dict] = None
    target_label: Optional[str] = None
    aggregate: str = "list"
    source: str = "rules"

    def to_json(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Answer:
    question: str
    plan: Plan
    items: List[dict]                  # [{"node", "score", "chains": [[Hop]]}]
    text: str = ""
    context: str = ""
    trace: List[dict] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    subgraph_nodes: List[str] = field(default_factory=list)
    llm_used: bool = False
    linked: List[str] = field(default_factory=list)


class GraphRAG:
    def __init__(self, kg: KnowledgeGraph, llm: Optional[LLM] = None):
        self.kg, self.llm = kg, llm
        docs = [e for e in kg.edges if e.get("evidence")]
        self._ev_edges = docs
        self._bm25 = BM25Okapi([self._tok(e["evidence"]) for e in docs]) if docs else None

    @staticmethod
    def _tok(s):
        return re.findall(r"[a-z0-9]+", strip_accents(s).lower())

    # ================================================================ 2a. lập kế hoạch theo luật
    def rule_plan(self, q: str, linked: List) -> Plan:
        qa = strip_accents(q).lower()
        ents = [l.node for l in linked]
        p = Plan(entities=ents, source="rules")
        if not ents:
            p.intent = "global" if re.search(GLOBAL_RE, qa) or re.search(TOP_RE, qa) else "none"
            return p
        if len(ents) >= 2 and re.search(PATH_RE, qa):
            p.intent = "path"; return p
        # vị trí thực thể đầu tiên trong câu (theo token)
        first_ent_pos = qa.find(strip_accents(linked[0].surface).split()[0]) if linked else 0
        hits = []
        for rx, rel, d in KEYWORDS:
            if rel is None:
                continue
            for m in re.finditer(rx, qa):
                # bỏ từ khoá nằm bên trong tên thực thể ("Nobel Prize")
                if any(strip_accents(l.surface) in qa[max(0, m.start() - 40):m.end() + 40] and
                       m.group(0) in strip_accents(l.surface) for l in linked):
                    continue
                hits.append((m.start(), rel, d))
        hits.sort()
        # loại trùng chồng lấn (cùng vị trí)
        dedup, last_end = [], -1
        for pos, rel, d in hits:
            if pos <= last_end:
                continue
            dedup.append((pos, rel, d)); last_end = pos + 3
        before = [h for h in dedup if h[0] < first_ent_pos]
        after = [h for h in dedup if h[0] >= first_ent_pos]
        ordered = list(reversed(before)) + after
        steps = []
        filter_award = None
        ent_labels = {e: self.kg.nodes[e]["label"] for e in ents}
        # thực thể thứ 2 là giải thưởng/viện/lĩnh vực -> dùng làm bộ lọc
        if len(ents) >= 2 and ent_labels[ents[1]] in ("Award", "Institution", "Field") and ent_labels[ents[0]] == "Scientist":
            rel_for = {"Award": "RECEIVED_AWARD", "Institution": "WORKED_AT", "Field": "WORKS_IN_FIELD"}[ent_labels[ents[1]]]
            if re.search(r"hoc tai|studied|tot nghiep", qa) and ent_labels[ents[1]] == "Institution":
                rel_for = "STUDIED_AT"
            filter_award = {"rel": rel_for, "target": ents[1]}
            ordered = [o for o in ordered if o[1] != rel_for]
        for pos, rel, d in ordered:
            if rel in ("CRITICIZED", "REFUTED", "INFLUENCED", "BUILT_UPON") and pos < first_ent_pos \
                    and re.search(WHO_RE, qa[:pos + 1]) and d == "out":
                d = "in"            # "Ai phê bình X?" -> X là đích của cạnh
            if rel == "ADVISOR_UP":
                steps.append({"rel": ["DOCTORAL_ADVISOR_OF", "ACADEMIC_ADVISOR_OF"], "dir": "in"})
            elif rel == "ADVISOR_DOWN":
                steps.append({"rel": ["DOCTORAL_ADVISOR_OF", "ACADEMIC_ADVISOR_OF"], "dir": "out"})
            else:
                steps.append({"rel": rel, "dir": d})
        # thực thể xuất phát là giải / viện / lĩnh vực: bước đầu đi ngược về nhà khoa học
        if ent_labels[ents[0]] in ("Award", "Institution", "Field"):
            back = {"Award": "RECEIVED_AWARD", "Institution": ["WORKED_AT", "STUDIED_AT"], "Field": "WORKS_IN_FIELD"}[ent_labels[ents[0]]]
            if re.search(r"hoc tai|studied|tot nghiep", qa) and ent_labels[ents[0]] == "Institution":
                back = "STUDIED_AT"
            elif ent_labels[ents[0]] == "Institution" and re.search(r"lam viec|worked|cong tac|giang day", qa):
                back = "WORKED_AT"
            steps = [s for s in steps if s["rel"] != back and s["rel"] != ["WORKED_AT", "STUDIED_AT"]]
            steps.insert(0, {"rel": back, "dir": "in"})
        if len(ents) >= 2 and not filter_award and re.search(COMMON_RE, qa):
            p.intent = "common"; p.steps = steps[:1] or [{"rel": None, "dir": "both"}]; return p
        if not steps:
            p.intent = "profile"; return p
        p.steps, p.filter = steps, filter_award
        if re.search(COUNT_RE, qa):
            p.aggregate = "count"
        elif re.search(TOP_RE, qa):
            p.aggregate = "top"
        return p

    # ================================================================ 2b. lập kế hoạch bằng LLM
    PLAN_PROMPT = """Bạn là bộ lập kế hoạch truy vấn cho một đồ thị tri thức về các nhà khoa học.
Loại nút: Scientist, Institution, Award, Field.
Các quan hệ (nguồn -> đích):
- DOCTORAL_ADVISOR_OF: thầy hướng dẫn tiến sĩ -> học trò
- ACADEMIC_ADVISOR_OF: cố vấn học thuật -> người được cố vấn
- INFLUENCED: người ảnh hưởng -> người chịu ảnh hưởng
- COLLABORATED_WITH, PRIORITY_DISPUTE: giữa hai nhà khoa học (dùng dir "both")
- CRITICIZED, REFUTED, BUILT_UPON: nhà khoa học -> nhà khoa học bị phê bình / bác bỏ / được kế thừa
- WORKED_AT, STUDIED_AT, ADVISOR_TO: nhà khoa học -> tổ chức
- RECEIVED_AWARD: nhà khoa học -> giải thưởng ; WORKS_IN_FIELD: nhà khoa học -> lĩnh vực
Trả về DUY NHẤT một JSON:
{"intent": "traverse|path|common|profile|global",
 "entities": [tên thực thể đúng như trong danh sách ỨNG VIÊN],
 "steps": [{"rel": "TÊN_QUAN_HỆ", "dir": "out|in|both"}],   // duyệt lần lượt bắt đầu từ entities[0]
 "filter": null hoặc {"rel": "...", "target": "tên thực thể"},  // giữ đáp án có cạnh rel tới target
 "aggregate": "list|count|top"}
"dir":"out" = đi theo chiều mũi tên từ nút hiện tại, "in" = đi ngược chiều.

Ví dụ:
Hỏi: Ai là thầy hướng dẫn của thầy hướng dẫn của Werner Heisenberg?
{"intent":"traverse","entities":["Werner Heisenberg"],"steps":[{"rel":"DOCTORAL_ADVISOR_OF","dir":"in"},{"rel":"DOCTORAL_ADVISOR_OF","dir":"in"}],"filter":null,"aggregate":"list"}
Hỏi: Những học trò của Arnold Sommerfeld đã đoạt giải Nobel Vật lý là ai?
{"intent":"traverse","entities":["Arnold Sommerfeld"],"steps":[{"rel":"DOCTORAL_ADVISOR_OF","dir":"out"}],"filter":{"rel":"RECEIVED_AWARD","target":"Nobel Prize in Physics"},"aggregate":"list"}
Hỏi: Học trò của Niels Bohr từng làm việc ở những nơi nào nhiều nhất?
{"intent":"traverse","entities":["Niels Bohr"],"steps":[{"rel":"DOCTORAL_ADVISOR_OF","dir":"out"},{"rel":"WORKED_AT","dir":"out"}],"filter":null,"aggregate":"top"}
Hỏi: Albert Einstein và Richard Feynman liên hệ với nhau qua những ai?
{"intent":"path","entities":["Albert Einstein","Richard Feynman"],"steps":[],"filter":null,"aggregate":"list"}
Hỏi: Max Born và Niels Bohr cùng nhận giải thưởng nào?
{"intent":"common","entities":["Max Born","Niels Bohr"],"steps":[{"rel":"RECEIVED_AWARD","dir":"out"}],"filter":null,"aggregate":"list"}
Hỏi: Những ai đã phê bình người bị Ernst Mach phê bình?
{"intent":"traverse","entities":["Ernst Mach"],"steps":[{"rel":"CRITICIZED","dir":"out"},{"rel":"CRITICIZED","dir":"in"}],"filter":null,"aggregate":"list"}
Hỏi: Các cộng đồng nghiên cứu lớn nhất trong mạng là gì?
{"intent":"global","entities":[],"steps":[],"filter":null,"aggregate":"top"}"""

    def llm_plan(self, q: str, linked: List) -> Optional[Plan]:
        if not (self.llm and self.llm.available):
            return None
        cands = [l.node for l in linked]
        msg = [{"role": "system", "content": self.PLAN_PROMPT},
               {"role": "user", "content": f"ỨNG VIÊN thực thể: {cands}\nHỏi: {q}"}]
        try:
            js = self.llm.chat_json(msg, max_tokens=300)
        except Exception as e:
            self.last_error = str(e)[:200]
            return None
        if not js:
            return None
        ents = []
        for e in js.get("entities", []) or []:
            if e in self.kg.nodes:
                ents.append(e)
            else:
                l = self.kg.link(str(e))
                if l:
                    ents.append(l[0].node)
        steps = []
        for st in js.get("steps", []) or []:
            rel = st.get("rel")
            rels = rel if isinstance(rel, list) else [rel]
            if all(r in RELS for r in rels) and st.get("dir", "out") in ("out", "in", "both"):
                steps.append({"rel": rel, "dir": st.get("dir", "out")})
            else:
                return None
        flt = js.get("filter")
        if flt:
            t = flt.get("target")
            if flt.get("rel") not in RELS or not t:
                flt = None
            elif t not in self.kg.nodes:
                l = self.kg.link(t); flt = {"rel": flt["rel"], "target": l[0].node} if l else None
        intent = js.get("intent", "traverse")
        if intent not in ("traverse", "path", "common", "profile", "global"):
            return None
        if intent in ("traverse", "common", "profile") and not ents:
            return None
        if intent == "path" and len(ents) < 2:
            return None
        return Plan(intent=intent, entities=ents, steps=steps, filter=flt,
                    aggregate=js.get("aggregate", "list") if js.get("aggregate") in ("list", "count", "top") else "list",
                    source=f"llm:{self.llm.model}")

    # ================================================================ 3. thực thi
    def execute(self, p: Plan):
        kg = self.kg
        if p.intent == "path":
            paths = kg.paths(p.entities[0], p.entities[1], k=3)
            items = [{"node": p.entities[1], "score": 1.0, "chains": paths}] if paths else []
            return items, [{"step": "shortest_paths", "found": len(paths),
                            "length": len(paths[0]) if paths else None}]
        if p.intent == "common":
            a, b = p.entities[:2]
            st = p.steps[0] if p.steps else {"rel": None, "dir": "both"}
            rels = st["rel"] if isinstance(st["rel"], list) else [st["rel"]]
            na = {(h.target if h.source == a else h.source): h for r in rels for h in kg.neighbors(a, r, st.get("dir", "both"))}
            nb = {(h.target if h.source == b else h.source): h for r in rels for h in kg.neighbors(b, r, st.get("dir", "both"))}
            common = [n for n in na if n in nb]
            items = [{"node": n, "score": float(kg.metrics.get(n, {}).get("pagerank", 0) or 0),
                      "chains": [[na[n]], [nb[n]]]} for n in common]
            items.sort(key=lambda x: -x["score"])
            return items, [{"step": "common_neighbours", "a": len(na), "b": len(nb), "common": len(common)}]
        if p.intent == "profile":
            hs = kg.facts(p.entities[0], 40)
            items = [{"node": (h.target if h.source == p.entities[0] else h.source), "score": 1.0, "chains": [[h]]} for h in hs]
            return items, [{"step": "profile", "facts": len(hs)}]
        if p.intent == "global":
            return [], [{"step": "community_summaries", "communities": len(kg.communities)}]
        frontier, trace = kg.traverse(p.entities[:1], p.steps)
        if p.filter:
            before = len(frontier)
            frontier = {n: c for n, c in frontier.items() if kg.has_edge_to(n, p.filter["rel"], p.filter["target"])}
            trace.append({"step": "filter", "rel": p.filter["rel"], "target": p.filter["target"],
                          "kept": len(frontier), "of": before})
        items = []
        cnt = getattr(kg, "last_counts", {})
        for n, chains in frontier.items():
            pr = float(kg.metrics.get(n, {}).get("pagerank", 0) or 0)
            npaths = cnt.get(n, len(chains))
            items.append({"node": n, "score": npaths + pr * 100, "n_paths": npaths, "chains": chains})
        if p.aggregate == "top":
            items.sort(key=lambda x: (-x["n_paths"], -x["score"]))
        else:
            items.sort(key=lambda x: -x["score"])
        return items, trace

    # ================================================================ 4. ngữ cảnh
    def evidence_search(self, q: str, k: int = 5) -> List[str]:
        if not self._bm25:
            return []
        sc = self._bm25.get_scores(self._tok(q))
        top = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [self._ev_edges[i]["evidence"] for i in top if sc[i] > 0]

    def build_context(self, p: Plan, items: List[dict], q: str) -> (str, List[str]):
        kg, lines, ev = self.kg, [], []
        lines.append(f"KẾ HOẠCH: {json.dumps(p.to_json(), ensure_ascii=False)}")
        if p.intent == "global":
            for c in kg.communities[:12]:
                s = kg.community_summaries.get(str(c["id"])) or (
                    f"Cộng đồng {c['id']} ({c['size']} nút): lĩnh vực {', '.join(c['top_fields'])}; "
                    f"viện {', '.join(c['top_institutions'][:2])}; nhà khoa học trung tâm {', '.join(c['top_scientists_pagerank'][:5])}; "
                    f"hạt nhân (k-core) {', '.join(c['nucleus'][:5])}.")
                lines.append(s)
            return "\n".join(lines), ev
        lines.append(f"SỐ ĐÁP ÁN: {len(items)}")
        for it in items[:15]:
            lab = kg.nodes.get(it["node"], {}).get("label", "")
            for ch in it["chains"][:2]:
                lines.append(f"- [{lab}] {it['node']} ⇐ " + " ; ".join(kg.describe(h) for h in ch))
                for h in ch:
                    if h.evidence and h.evidence not in ev:
                        ev.append(h.evidence)
        for e in p.entities[:2]:
            m = kg.metrics.get(e, {})
            facts = kg.facts(e, 12)
            lines.append(f"DỮ KIỆN VỀ {e}: " + "; ".join(kg.describe(h) for h in facts) +
                         (f". PageRank hạng {m.get('pagerank_rank')}, cộng đồng {m.get('community')}, k-core {m.get('core_number')}" if m else ""))
        ev += [s for s in self.evidence_search(q + " " + " ".join(p.entities), 3) if s not in ev]
        if ev:
            lines.append("CÂU BẰNG CHỨNG (Wikipedia): " + " | ".join(ev[:6]))
        return "\n".join(lines), ev[:8]

    # ================================================================ 5. sinh câu trả lời
    ANSWER_PROMPT = ("Bạn là trợ lý phân tích mạng xã hội các nhà khoa học. Trả lời bằng tiếng Việt, ngắn gọn, "
                     "CHỈ dựa vào NGỮ CẢNH (kết quả duyệt đồ thị). Nêu rõ chuỗi suy luận theo dạng A → B → C. "
                     "Nếu ngữ cảnh không có đáp án thì nói không tìm thấy trong đồ thị, không bịa thêm. "
                     "Giữ nguyên tên riêng tiếng Anh.")

    def template_answer(self, p: Plan, items: List[dict]) -> str:
        kg = self.kg
        if p.intent == "none":
            return "Tôi chưa nhận ra nhà khoa học / giải thưởng / tổ chức nào trong câu hỏi. Hãy ghi rõ tên (vd: Niels Bohr)."
        if p.intent == "global":
            cs = kg.communities[:6]
            return "Các cộng đồng lớn nhất (Louvain):\n\n" + "\n".join(
                f"- Cộng đồng {c['id']} – {c['size']} nút, lĩnh vực {', '.join(c['top_fields'][:2])}; "
                f"trung tâm: {', '.join(c['top_scientists_pagerank'][:4])}" for c in cs)
        if not items:
            return "Không tìm thấy đáp án trong đồ thị cho kế hoạch truy vấn này."
        if p.intent == "profile":
            e = p.entities[0]; m = kg.metrics.get(e, {})
            groups = {}
            for it in items:
                h = it["chains"][0][0]
                fw, bw = REL_VI.get(h.rel, (h.rel, h.rel))
                lab = fw if h.source == e else bw
                groups.setdefault(lab, []).append(it["node"])
            lines = [f"**{e}** ({kg.nodes[e]['label']})"]
            if m:
                lines.append(f"PageRank hạng {m.get('pagerank_rank')} / {len(kg.nodes)}, bậc {m.get('degree')}, "
                             f"cộng đồng {m.get('community')}, k-core {m.get('core_number')}")
            for lab, ns in groups.items():
                lines.append(f"- {lab}: " + ", ".join(ns[:8]) + (f" … (+{len(ns) - 8})" if len(ns) > 8 else ""))
            return lines[0] + "  \n" + lines[1] + "\n\n" + "\n".join(lines[2:]) if len(lines) > 1 else lines[0]
        if p.intent == "path":
            ch = items[0]["chains"][0]
            return (f"{p.entities[0]} và {p.entities[1]} cách nhau {len(ch)} bước:\n\n" +
                    "\n".join(f"{i + 1}. {kg.describe(h)}" for i, h in enumerate(ch)))
        if p.aggregate == "count":
            return f"Có {len(items)} kết quả: " + ", ".join(i["node"] for i in items[:15]) + ("…" if len(items) > 15 else "")
        if p.aggregate == "top":
            return "Xếp theo số đường suy luận dẫn tới:\n\n" + "\n".join(
                f"{k + 1}. {i['node']} ({i['n_paths']} đường)" for k, i in enumerate(items[:10]))
        head = f"Tìm thấy {len(items)} kết quả" + (":" if len(items) <= 12 else " (12 nổi bật nhất):")
        lines = []
        for i in items[:12]:
            ch = i["chains"][0]
            lines.append(f"- **{i['node']}** — " + " → ".join(kg.describe(h) for h in ch))
        return head + "\n\n" + "\n".join(lines)

    def ask(self, q: str, use_llm: bool = True) -> Answer:
        linked = self.kg.link(q)
        plan = self.llm_plan(q, linked) if use_llm else None
        if plan is None:
            plan = self.rule_plan(q, linked)
        items, trace = self.execute(plan) if plan.intent != "none" else ([], [])
        ctx, ev = self.build_context(plan, items, q) if plan.intent != "none" else ("", [])
        ans = Answer(q, plan, items, context=ctx, trace=trace, evidence=ev, linked=[l.node for l in linked])
        sub = set(plan.entities)
        for it in items[:12]:
            for ch in it["chains"][:2]:
                for h in ch:
                    sub.update([h.source, h.target])
        ans.subgraph_nodes = list(sub)
        if use_llm and self.llm and self.llm.available and plan.intent != "none":
            try:
                ans.text = self.llm.chat([{"role": "system", "content": self.ANSWER_PROMPT},
                                          {"role": "user", "content": f"NGỮ CẢNH:\n{ctx}\n\nCÂU HỎI: {q}"}],
                                         max_tokens=600)
                ans.llm_used = True
            except Exception as e:
                self.last_error = str(e)[:200]
        if not ans.text:
            ans.text = self.template_answer(plan, items)
        return ans
