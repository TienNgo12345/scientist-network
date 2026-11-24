"""Bước 11 – Trích quan hệ theo vai trò ngữ pháp (RoleExtractor) trên kho văn bản đã làm giàu NER.

Khác bước 2: (a) chạy trên MỌI mục văn bản thay vì chỉ Career/Research/Controversies,
(b) thực thể không chỉ là wikilink mà gồm cả mention do mô hình NER + EL (bước 9) phát hiện
(ví dụ lần nhắc thứ hai "Bohr" không có link vẫn được nhận ra là Niels Bohr).

Đầu ra: data/enrich/rule_new_edges.csv (chỉ các cạnh CHƯA có trong đồ thị)
Chạy  : python scripts/11_enrich_rules.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm

from scinet.config import DATA
from scinet.relation_extractor import RoleExtractor, to_dicts
from scinet.wikitext import LinkSpan, Section

ENRICH = DATA / "enrich"
OUT = DATA / "output"


def main():
    labels = {r["name"]: r["label"] for r in csv.DictReader(open(OUT / "nodes.csv", encoding="utf-8"))}
    existing = set()
    for r in csv.DictReader(open(OUT / "edges.csv", encoding="utf-8")):
        existing.add((r["source"], r["target"], r["type"]))
        if r["type"] == "COLLABORATED_WITH":
            existing.add((r["target"], r["source"], r["type"]))
    meta = [json.loads(l) for l in open(ENRICH / "sections.jsonl", encoding="utf-8")]
    ments = [json.loads(l) for l in open(ENRICH / "mentions.jsonl", encoding="utf-8")]
    by_page = {}
    for m, mm in zip(meta, ments):
        spans = [LinkSpan(x["start"], x["end"], x["kb"], x["text"]) for x in mm["mentions"] if x["kb"]]
        by_page.setdefault(m["page"], []).append(Section(m["heading"], m["text"], spans))
    ex = RoleExtractor()
    rows = []
    for page, secs in tqdm(by_page.items(), desc="rules"):
        try:
            rels = ex.extract(page, secs, labels)
        except Exception:
            continue
        for d in to_dicts(rels):
            key = (d["source"], d["target"], d["rel_type"])
            if key in existing or d["source"] == d["target"]:
                continue
            d["method"] = "rule+role+ner"
            rows.append(d)
    seen, out = set(), []
    for d in rows:
        k = (d["source"], d["target"], d["rel_type"])
        if k not in seen:
            seen.add(k); out.append(d)
    with open(ENRICH / "rule_new_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "type", "prob", "evidence", "page",
                                          "section", "trigger", "method"])
        w.writeheader()
        for d in out:
            w.writerow({"source": d["source"], "target": d["target"], "type": d["rel_type"],
                        "prob": d.get("confidence", 0.85), "evidence": d.get("evidence", ""),
                        "page": d.get("page", ""), "section": d.get("section", ""),
                        "trigger": d.get("trigger", ""), "method": d["method"]})
    print(len(out), "cạnh mới:", dict(Counter(d["rel_type"] for d in out)))


if __name__ == "__main__":
    main()
