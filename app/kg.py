"""Đồ thị tri thức trong bộ nhớ (NetworkX) + liên kết thực thể trong câu hỏi + duyệt nhiều bước.

Ứng dụng đọc thẳng data/final/*.csv nên chạy được ngay cả khi không có Neo4j.
(Có thể đổi sang Neo4j bằng app/neo4j_backend.py nếu muốn.)
"""
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "final"

REL_VI = {
    "DOCTORAL_ADVISOR_OF": ("hướng dẫn tiến sĩ cho", "có thầy hướng dẫn tiến sĩ là"),
    "ACADEMIC_ADVISOR_OF": ("cố vấn học thuật cho", "có cố vấn học thuật là"),
    "INFLUENCED": ("ảnh hưởng tới", "chịu ảnh hưởng của"),
    "COLLABORATED_WITH": ("hợp tác với", "hợp tác với"),
    "CRITICIZED": ("phê bình", "bị phê bình bởi"),
    "REFUTED": ("bác bỏ", "bị bác bỏ bởi"),
    "BUILT_UPON": ("kế thừa công trình của", "được kế thừa bởi"),
    "PRIORITY_DISPUTE": ("tranh chấp quyền tác giả với", "tranh chấp quyền tác giả với"),
    "WORKED_AT": ("làm việc tại", "là nơi làm việc của"),
    "STUDIED_AT": ("học tại", "là nơi học của"),
    "ADVISOR_TO": ("làm cố vấn cho", "có cố vấn là"),
    "RECEIVED_AWARD": ("nhận giải", "được trao cho"),
    "WORKS_IN_FIELD": ("nghiên cứu lĩnh vực", "là lĩnh vực của"),
}
LABEL_COLOR = {"Scientist": "#4C78A8", "Institution": "#54A24B", "Award": "#E45756", "Field": "#F2A93B"}

# Tên tiếng Việt thường gặp trong câu hỏi -> nút trong đồ thị
VI_ALIASES = {
    "nobel vật lý": "Nobel Prize in Physics", "nobel vật lí": "Nobel Prize in Physics",
    "nobel hóa học": "Nobel Prize in Chemistry", "nobel hoá học": "Nobel Prize in Chemistry",
    "nobel y học": "Nobel Prize in Physiology or Medicine", "nobel sinh lý": "Nobel Prize in Physiology or Medicine",
    "nobel kinh tế": "Nobel Memorial Prize in Economic Sciences", "huy chương fields": "Fields Medal",
    "giải turing": "Turing Award", "giải abel": "Abel Prize", "huy chương copley": "Copley Medal",
    "giải wolf vật lý": "Wolf Prize in Physics", "giải kyoto": "Kyoto Prize",
    "vật lý lý thuyết": "Theoretical physics", "vật lý": "Physics", "vật lí": "Physics", "hóa học": "Chemistry",
    "hoá học": "Chemistry", "toán học": "Mathematics", "kinh tế học": "Economics", "sinh học": "Biology",
    "khoa học máy tính": "Computer science", "thiên văn học": "Astronomy", "y học": "Medicine",
    "đại học harvard": "Harvard University", "đại học cambridge": "University of Cambridge",
    "đại học oxford": "University of Oxford", "đại học princeton": "Princeton University",
    "đại học göttingen": "University of Göttingen", "đại học gottingen": "University of Göttingen",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def key(s: str) -> str:
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    return re.sub(r"[^a-z0-9 ]", " ", strip_accents(s).lower()).split().__str__()


@dataclass
class Hop:
    source: str
    rel: str
    target: str
    evidence: str = ""
    method: str = ""

    def text(self) -> str:
        return f"{self.source} —[{self.rel}]→ {self.target}"


@dataclass
class Linked:
    node: str
    surface: str
    score: float
    start: int = -1


class KnowledgeGraph:
    def __init__(self, folder: Path = FINAL):
        self.nodes: Dict[str, dict] = {r["name"]: r for r in csv.DictReader(open(folder / "nodes.csv", encoding="utf-8"))}
        self.G = nx.MultiDiGraph()
        for n, r in self.nodes.items():
            self.G.add_node(n, **r)
        self.edges = list(csv.DictReader(open(folder / "edges.csv", encoding="utf-8")))
        for e in self.edges:
            self.G.add_edge(e["source"], e["target"], key=e["type"], type=e["type"],
                            evidence=e.get("evidence", ""), method=e.get("method", ""))
        self.U = nx.Graph(self.G.to_undirected())
        self.metrics = {}
        mp = folder / "node_metrics.csv"
        if mp.exists():
            for r in csv.DictReader(open(mp, encoding="utf-8")):
                self.metrics[r["name"]] = r
        cp = folder / "communities.json"
        self.communities = json.load(open(cp, encoding="utf-8")) if cp.exists() else []
        sp = folder / "community_summaries.json"
        self.community_summaries = json.load(open(sp, encoding="utf-8")) if sp.exists() else {}
        self._build_index()

    # ------------------------------------------------------------------ liên kết thực thể
    def _build_index(self):
        self.index: Dict[str, Counter] = defaultdict(Counter)

        def add(surface, node, w):
            k = " ".join(re.sub(r"[^a-z0-9 ]", " ", strip_accents(surface).lower()).split())
            if len(k) >= 3:
                self.index[k][node] += w
        deg = {n: self.U.degree(n) for n in self.nodes}
        for n in self.nodes:
            add(n, n, 10 + deg[n] / 100)
            base = re.sub(r"\s*\(.*?\)\s*$", "", n)
            add(base, n, 8 + deg[n] / 100)
        al = FINAL / "aliases.json"
        if not al.exists():
            al = ROOT / "data" / "interim" / "aliases.json"
        if al.exists():
            for a, t in json.load(open(al, encoding="utf-8")).items():
                if t in self.nodes:
                    add(a, t, 5)
        # họ của nhà khoa học (chỉ khi duy nhất, ưu tiên người nổi bật)
        sur = defaultdict(list)
        for n, r in self.nodes.items():
            if r["label"] == "Scientist":
                base = re.sub(r"\s*\(.*?\)\s*$", "", n)
                sur[base.split()[-1]].append(n)
        for s, ns in sur.items():
            best = max(ns, key=lambda n: deg[n])
            if len(ns) == 1 or deg[best] >= 3 * sorted([deg[x] for x in ns])[-2]:
                add(s, best, 1)
        for vi, n in VI_ALIASES.items():
            if n in self.nodes:
                add(vi, n, 50)
        self._names = list(self.nodes)

    def link(self, text: str, max_n: int = 6) -> List[Linked]:
        """Tìm thực thể trong câu hỏi: khớp n-gram dài nhất trước, sau đó khớp mờ cho tên viết sai."""
        raw = strip_accents(text).lower()
        toks = re.findall(r"[a-z0-9]+", raw)
        used = [False] * len(toks)
        found: List[Linked] = []
        for n in range(min(max_n, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(used[i:i + n]):
                    continue
                g = " ".join(toks[i:i + n])
                if g in self.index:
                    if n == 1 and (len(g) < 4 or g in STOP_WORDS):
                        continue
                    node, sc = self.index[g].most_common(1)[0]
                    found.append(Linked(node, g, sc, i))
                    for j in range(i, i + n):
                        used[j] = True
        if not found:
            # khớp mờ trên các cụm viết hoa trong câu gốc (VD "Heisenburg")
            for m in re.finditer(r"([A-ZÀ-Ỹ][\w\-'.]+(?:\s+[A-ZÀ-Ỹ][\w\-'.]+){0,3})", text):
                cand = m.group(1)
                if strip_accents(cand).lower() in STOP_WORDS:
                    continue
                best = process.extractOne(cand, self._names, scorer=fuzz.WRatio, score_cutoff=88)
                if best:
                    found.append(Linked(best[0], cand, best[1] / 10, m.start()))
        found.sort(key=lambda x: x.start)
        out, seen = [], set()
        for f in found:
            if f.node not in seen:
                seen.add(f.node); out.append(f)
        return out

    def search(self, q: str, label: Optional[str] = None, limit: int = 10) -> List[str]:
        names = [n for n in self._names if label is None or self.nodes[n]["label"] == label]
        res = process.extract(q, names, scorer=fuzz.WRatio, limit=limit * 3)
        res = sorted(res, key=lambda x: (-x[1], -float(self.metrics.get(x[0], {}).get("pagerank", 0) or 0)))
        return [r[0] for r in res[:limit]]

    # ------------------------------------------------------------------ duyệt đồ thị
    def neighbors(self, n: str, rel: Optional[str] = None, direction: str = "out") -> List[Hop]:
        out = []
        if direction in ("out", "both") and n in self.G:
            for _, t, d in self.G.out_edges(n, data=True):
                if rel is None or d["type"] == rel:
                    out.append(Hop(n, d["type"], t, d.get("evidence", ""), d.get("method", "")))
        if direction in ("in", "both") and n in self.G:
            for s, _, d in self.G.in_edges(n, data=True):
                if rel is None or d["type"] == rel:
                    out.append(Hop(s, d["type"], n, d.get("evidence", ""), d.get("method", "")))
        return out

    def traverse(self, starts: List[str], steps: List[dict], max_frontier: int = 3000):
        """Duyệt nhiều bước. steps = [{"rel": REL|None, "dir": "out"|"in"|"both"}, ...]
        Trả về {nút_cuối: [chuỗi Hop từ nút xuất phát]} (giữ tối đa 3 chuỗi cho mỗi nút)."""
        frontier = {s: [[]] for s in starts if s in self.G}
        counts = {s: 1 for s in frontier}           # số đường (không giới hạn) tới mỗi nút
        trace = []
        for k, st in enumerate(steps):
            nxt: Dict[str, List[List[Hop]]] = defaultdict(list)
            ncount: Dict[str, int] = defaultdict(int)
            rels = st.get("rel")
            rels = rels if isinstance(rels, list) else [rels]
            for node, chains in frontier.items():
                for r in rels:
                    for h in self.neighbors(node, r, st.get("dir", "out")):
                        other = h.target if h.source == node else h.source
                        if other in starts and k > 0:
                            continue
                        ncount[other] += counts.get(node, 1)
                        for c in chains[:3]:
                            if len(nxt[other]) < 3:
                                nxt[other].append(c + [h])
            trace.append({"step": k + 1, "rel": st.get("rel"), "dir": st.get("dir", "out"), "reached": len(nxt)})
            frontier = dict(list(nxt.items())[:max_frontier])
            counts = {n: ncount[n] for n in frontier}
            if not frontier:
                break
        self.last_counts = counts
        return frontier, trace

    def has_edge_to(self, n: str, rel: str, target: str) -> bool:
        return any(h.target == target for h in self.neighbors(n, rel, "out"))

    def paths(self, a: str, b: str, k: int = 3, cutoff: int = 8, academic_first: bool = True) -> List[List[Hop]]:
        """k đường ngắn nhất giữa a và b (vô hướng), kèm loại quan hệ và hướng thật của từng cạnh.
        academic_first: thử trước trên đồ thị bỏ nút Award/Field (đường qua 'giải Nobel' ít thông tin)."""
        if a not in self.U or b not in self.U:
            return []
        if academic_first:
            if not hasattr(self, "_Uacad"):
                keep = [n for n, r in self.nodes.items() if r["label"] in ("Scientist", "Institution")]
                self._Uacad = self.U.subgraph(keep)
            res = self._paths_in(self._Uacad, a, b, k, cutoff)
            if res:
                return res
        return self._paths_in(self.U, a, b, k, cutoff)

    def _paths_in(self, U, a, b, k, cutoff):
        if a not in U or b not in U:
            return []
        out = []
        try:
            gen = nx.shortest_simple_paths(U, a, b)
            for p in gen:
                if len(p) - 1 > cutoff or len(out) >= k:
                    break
                hops = []
                for u, v in zip(p, p[1:]):
                    hs = [h for h in self.neighbors(u, None, "both") if {h.source, h.target} == {u, v}]
                    hops.append(hs[0])
                out.append(hops)
        except nx.NetworkXNoPath:
            pass
        return out

    def lineage(self, n: str, up: bool = True, depth: int = 6, academic: bool = True) -> List[List[str]]:
        """Chuỗi thầy (up) hoặc trò (down) theo DOCTORAL_ADVISOR_OF (+ ACADEMIC_ADVISOR_OF nếu academic)."""
        rels = ["DOCTORAL_ADVISOR_OF"] + (["ACADEMIC_ADVISOR_OF"] if academic else [])
        chains, cur = [], [[n]]
        for _ in range(depth):
            nxt = []
            for c in cur:
                hs = [h for r in rels for h in self.neighbors(c[-1], r, "in" if up else "out")]
                for h in hs:
                    o = h.source if up else h.target
                    if o not in c:
                        nxt.append(c + [o])
            if not nxt:
                break
            chains += nxt; cur = nxt[:200]
        return chains

    def facts(self, n: str, limit: int = 40) -> List[Hop]:
        hs = self.neighbors(n, None, "both")
        pri = {"DOCTORAL_ADVISOR_OF": 0, "ACADEMIC_ADVISOR_OF": 1, "INFLUENCED": 2, "CRITICIZED": 2, "REFUTED": 2,
               "BUILT_UPON": 2, "COLLABORATED_WITH": 3, "PRIORITY_DISPUTE": 2, "RECEIVED_AWARD": 4, "WORKED_AT": 5,
               "STUDIED_AT": 5, "WORKS_IN_FIELD": 6, "ADVISOR_TO": 6}
        return sorted(hs, key=lambda h: pri.get(h.rel, 9))[:limit]

    def ego(self, centers: List[str], radius: int = 1, max_nodes: int = 70) -> nx.MultiDiGraph:
        keep = set(centers)
        frontier = set(centers)
        for _ in range(radius):
            nxt = set()
            for c in frontier:
                nb = sorted(set(self.U.neighbors(c)) if c in self.U else [],
                            key=lambda x: -float(self.metrics.get(x, {}).get("pagerank", 0) or 0))
                nxt.update(nb[: max(5, max_nodes // max(1, len(frontier)))])
            keep |= nxt; frontier = nxt
            if len(keep) >= max_nodes:
                break
        return self.G.subgraph(list(keep)[:max_nodes]).copy()

    def describe(self, h: Hop) -> str:
        fw, _ = REL_VI.get(h.rel, (h.rel, h.rel))
        return f"{h.source} {fw} {h.target}"

    def community_of(self, n: str) -> Optional[dict]:
        c = self.metrics.get(n, {}).get("community")
        if c is None or c == "":
            return None
        c = int(c)
        return next((x for x in self.communities if x["id"] == c), None)


STOP_WORDS = {"who", "what", "which", "where", "when", "the", "and", "with", "was", "were", "did", "does",
              "nobel", "prize", "award", "university", "ai", "nao", "nhung", "cua", "la", "va", "co", "khong",
              "thay", "tro", "hoc", "nguoi", "giai", "lam", "viec", "tai", "o", "dau", "bao", "nhieu", "cung",
              "huong", "dan", "voi", "nhau", "duoc", "nhat", "tung", "da", "trong", "cac", "mot", "nhu", "the",
              "school", "institute", "college", "physics", "law", "field", "fields", "medal", "order", "science",
              "student", "students", "advisor", "work", "worked", "influence", "trinh", "mang", "lien", "he"}
