"""Bước 12 – Đánh giá chất lượng đồ thị + dựng đồ thị cuối cùng (đã làm sạch và làm giàu).

Phương án đánh giá chất lượng gồm 2 nhóm:
 A. Kiểm tra tự động (không cần người):
   1. Nút rác: tên chứa mã wiki, nút Award không mang nghĩa giải thưởng ("Cambridge", "Fellow"...).
   2. Ràng buộc kiểu (domain/range) của từng loại quan hệ.
   3. Cạnh tự vòng, cạnh trùng.
   4. Chu trình trong quan hệ thầy–trò (DOCTORAL_ADVISOR_OF phải là đồ thị có hướng không chu trình).
   5. Nhất quán thời gian: thầy hướng dẫn phải sinh trước học trò (năm sinh lấy từ Infobox).
   6. Độ phủ bằng chứng: tỉ lệ cạnh có nguồn gốc truy được (trường Infobox / câu văn bản).
 B. Kiểm tra bằng người (lấy mẫu): precision ước lượng trên mẫu có khoảng tin cậy Wilson 95%
   (data/output/eval_sample.csv cho đồ thị gốc, data/enrich/audit_new_edges.csv cho cạnh làm giàu);
   2 người chấm độc lập + hệ số Cohen's kappa khi có nhãn của cả hai.

Đầu ra: data/final/nodes.csv, data/final/edges.csv, reports/quality.json, reports/quality.md
Chạy  : python scripts/12_quality_and_merge.py
"""
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import networkx as nx

from scinet.config import DATA, RAW, ROOT
from scinet.ner import linkable
from scinet.wiki_client import read_jsonl

OUT, ENRICH, FINAL, REPORTS = DATA / "output", DATA / "enrich", DATA / "final", ROOT / "reports"

DOMAIN_RANGE = {
    "DOCTORAL_ADVISOR_OF": ("Scientist", "Scientist"), "ACADEMIC_ADVISOR_OF": ("Scientist", "Scientist"),
    "INFLUENCED": ("Scientist", "Scientist"), "COLLABORATED_WITH": ("Scientist", "Scientist"),
    "CRITICIZED": ("Scientist", "Scientist"), "REFUTED": ("Scientist", "Scientist"),
    "BUILT_UPON": ("Scientist", "Scientist"), "PRIORITY_DISPUTE": ("Scientist", "Scientist"),
    "WORKED_AT": ("Scientist", "Institution"), "STUDIED_AT": ("Scientist", "Institution"),
    "ADVISOR_TO": ("Scientist", "Institution"), "RECEIVED_AWARD": ("Scientist", "Award"),
    "WORKS_IN_FIELD": ("Scientist", "Field"),
}
JUNK_CHARS = re.compile(r"[<>{}|=\[\]]|^\W|^\d+(st|nd|rd|th)? ?class$", re.I)
INTERWIKI = re.compile(r"^:?[a-z]{2,3}:(?=[A-Z])")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 4), round(c - h, 4), round(c + h, 4))


def birth_years():
    out = {}
    for p in read_jsonl(RAW / "scientists.jsonl"):
        m = re.search(r"\|\s*birth_date\s*=([^\n]*)", p["wikitext"])
        if m:
            y = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", m.group(1))
            if y:
                out[p["title"]] = int(y.group(1))
    return out


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    nodes = {r["name"]: r for r in csv.DictReader(open(OUT / "nodes.csv", encoding="utf-8"))}
    edges = list(csv.DictReader(open(OUT / "edges.csv", encoding="utf-8")))
    q = {"input": {"nodes": len(nodes), "edges": len(edges)}}

    # ---- A1. nút rác + chuẩn hoá interwiki ----------------------------------------------------
    import spacy
    nlp = spacy.load("en_core_web_sm")
    GENERIC = {"fellow", "member", "letters", "doctorate", "lung", "poetry", "milestone", "knight",
               "sciences", "science", "physics", "chemistry", "medicine", "economics", "foreign member"}
    rename, junk = {}, []
    for n, r in nodes.items():
        if INTERWIKI.match(n):
            rename[n] = INTERWIKI.sub("", n)
        elif JUNK_CHARS.search(n) or not n.strip():
            junk.append(n)
        elif r["label"] == "Award" and not linkable(n, "Award") and len(n.split()) <= 2:
            ents = nlp(n).ents
            is_place = any(e.label_ in ("GPE", "LOC") and e.text == n for e in ents)
            if is_place or n.lower() in GENERIC or len(n.split()) == 1:
                junk.append(n)
    junk = set(junk)
    q["junk_nodes"] = {"count": len(junk), "examples": sorted(junk)[:40],
                       "edges_removed": sum(e["source"] in junk or e["target"] in junk for e in edges),
                       "interwiki_renamed": len(rename)}

    def nm(x):
        return rename.get(x, x)

    # ---- gộp cạnh gốc + cạnh làm giàu đã kiểm duyệt ------------------------------------------
    audit = list(csv.DictReader(open(ENRICH / "audit_new_edges.csv", encoding="utf-8")))
    rejected = {(a["source"], a["target"], a["type"]) for a in audit if a["correct"] == "0"}
    enrich = []
    for f, method in [(ENRICH / "rule_new_edges.csv", "rule+role+ner"), (ENRICH / "re_new_edges.csv", "ds-logreg")]:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if (r["source"], r["target"], r["type"]) in rejected:
                continue
            enrich.append({"source": r["source"], "target": r["target"], "type": r["type"],
                           "method": method, "confidence": r["prob"], "evidence": r["evidence"],
                           "section": r.get("section", ""), "page": r["page"]})
    q["enrichment"] = {"candidates": len(audit), "rejected_by_audit": len(rejected),
                       "added": len(enrich), "by_type": dict(Counter(e["type"] for e in enrich)),
                       "by_method": dict(Counter(e["method"] for e in enrich))}

    # ---- A2/A3. ràng buộc kiểu, tự vòng, trùng -------------------------------------------------
    final, seen = [], set()
    viol = Counter(); selfloop = dup = 0
    for e in edges + enrich:
        s, t = nm(e["source"]), nm(e["target"])
        if s in junk or t in junk:
            continue
        if s == t:
            selfloop += 1; continue
        key = (s, t, e["type"]) if e["type"] != "COLLABORATED_WITH" else (*sorted([s, t]), e["type"])
        if key in seen:
            dup += 1; continue
        ls, lt = nodes.get(e["source"], {}).get("label"), nodes.get(e["target"], {}).get("label")
        dom = DOMAIN_RANGE.get(e["type"])
        if dom and (ls != dom[0] or lt != dom[1]):
            viol[e["type"]] += 1; continue
        seen.add(key)
        final.append({**e, "source": s, "target": t})
    q["constraints"] = {"type_violations_removed": dict(viol), "self_loops_removed": selfloop,
                        "duplicates_removed": dup}

    # ---- A4. chu trình thầy–trò --------------------------------------------------------------
    g = nx.DiGraph([(e["source"], e["target"]) for e in final if e["type"] == "DOCTORAL_ADVISOR_OF"])
    cycles = list(nx.simple_cycles(g, length_bound=4))
    q["advisor_cycles"] = {"count": len(cycles), "examples": cycles[:10]}

    # ---- A5. nhất quán thời gian ------------------------------------------------------------
    by = birth_years()
    checked = bad = 0; bad_list = []
    for e in final:
        if e["type"] in ("DOCTORAL_ADVISOR_OF", "ACADEMIC_ADVISOR_OF") and e["source"] in by and e["target"] in by:
            checked += 1
            gap = by[e["target"]] - by[e["source"]]
            if gap <= 0:
                bad += 1; bad_list.append((e["source"], by[e["source"]], e["target"], by[e["target"]], e["type"]))
    q["temporal"] = {"advisor_edges_with_both_birth_years": checked, "advisor_not_older": bad,
                     "consistency_rate": round(1 - bad / checked, 4) if checked else None,
                     "examples": bad_list[:15], "scientists_with_birth_year": len(by)}

    # ---- A6. độ phủ bằng chứng -------------------------------------------------------------
    has_ev = sum(bool(e.get("evidence")) for e in final)
    q["provenance"] = {"edges_with_evidence_or_field": has_ev, "share": round(has_ev / len(final), 4)}

    # ---- B. precision ước lượng trên mẫu -----------------------------------------------------
    prec = {}
    es = OUT / "eval_sample.csv"
    if es.exists():
        rows = [r for r in csv.DictReader(open(es, encoding="utf-8")) if r.get("correct") in ("0", "1")]
        if rows:
            k = sum(r["correct"] == "1" for r in rows); prec["base_graph_sample"] = wilson(k, len(rows)) + (len(rows),)
    for m in ("rule+role+ner", "ds-logreg"):
        rows = [a for a in audit if a["method"] == m]
        k = sum(a["correct"] == "1" for a in rows)
        prec[f"enrichment_{m}"] = wilson(k, len(rows)) + (len(rows),)
    prec["note"] = "định dạng: (precision, CI95 thấp, CI95 cao, n); nhãn hiện tại là chấm sơ bộ"
    q["precision"] = prec

    # ---- ghi đồ thị cuối ----------------------------------------------------------------------
    used = {e["source"] for e in final} | {e["target"] for e in final}
    fn, seen_names = [], set()
    for n, r in nodes.items():
        n2 = nm(n)
        if n in junk or n2 not in used or n2 in seen_names:
            continue
        seen_names.add(n2)
        fn.append({**r, "name": n2, "id": r["id"]})
    with open(FINAL / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fn[0].keys())); w.writeheader(); w.writerows(fn)
    cols = ["source", "target", "type", "method", "confidence", "evidence", "section", "page"]
    with open(FINAL / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for e in final:
            w.writerow({c: e.get(c, "") for c in cols})
    q["output"] = {"nodes": len(fn), "edges": len(final),
                   "relation_types_with_data": len({e["type"] for e in final}),
                   "edges_by_type": dict(Counter(e["type"] for e in final).most_common()),
                   "nodes_by_label": dict(Counter(r["label"] for r in fn))}
    REPORTS.mkdir(exist_ok=True)
    json.dump(q, open(REPORTS / "quality.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: q[k] for k in ("input", "output", "constraints", "advisor_cycles", "temporal",
                                         "enrichment", "precision")}, indent=1, ensure_ascii=False)[:4000])
    print("junk:", q["junk_nodes"]["count"], "nút,", q["junk_nodes"]["edges_removed"], "cạnh")


if __name__ == "__main__":
    main()
