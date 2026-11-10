"""Bước 2 – Tiền xử lý + trích xuất quan hệ (Infobox + NLP).

Đầu vào : data/raw/scientists.jsonl
Đầu ra  : data/interim/infobox_edges.jsonl, data/interim/nlp_edges.jsonl,
          data/interim/link_types.json (loại của các trang được link tới)
Chạy    : python scripts/02_extract.py [--nli facebook/bart-large-mnli] [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm

from scinet.config import (RAW, INTERIM, INFOBOX_FIELD_MAP, SCIENTIST_INFOBOXES,
                           INSTITUTION_INFOBOXES)
from scinet.wikitext import parse_infobox, sections_with_links
from scinet.relation_extractor import RelationExtractor, RoleExtractor, to_dicts
from scinet.wiki_client import WikiClient, read_jsonl, classify_titles


def infobox_edges(page: dict):
    title = page["title"]
    name, box = parse_infobox(page["wikitext"])
    edges = []
    for field, val in box.items():
        if field not in INFOBOX_FIELD_MAP:
            continue
        rtype, direction, ttype = INFOBOX_FIELD_MAP[field]
        targets = [(t, "link") for t in val["links"]] + [(t, "plain") for t in val["plain"]]
        for t, kind in targets:
            # Người / tổ chức phải là link để đảm bảo độ chính xác; Field/Award chấp nhận text
            if kind == "plain" and ttype in ("Scientist", "Institution"):
                continue
            src, dst = (title, t) if direction == "out" else (t, title)
            edges.append({"source": src, "target": dst, "rel_type": rtype,
                          "target_type": ttype, "field": field, "is_link": kind == "link",
                          "page": title, "method": "infobox",
                          "confidence": 1.0 if kind == "link" else 0.9})
    return name, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", default=None, help="vd facebook/bart-large-mnli (tuỳ chọn)")
    ap.add_argument("--model", default="en_core_web_sm")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--legacy", action="store_true", help="dùng bộ trích xuất v1 (regex + hướng)")
    ap.add_argument("--offline", action="store_true", help="không gọi API phân loại link")
    args = ap.parse_args()

    pages = list(read_jsonl(RAW / "scientists.jsonl"))[: args.limit]
    scientist_titles = {p["title"] for p in pages}
    print(f"{len(pages)} trang nhà khoa học")

    # --- 2a. Infobox --------------------------------------------------------
    ib_rows, box_names = [], {}
    for p in tqdm(pages, desc="infobox"):
        name, e = infobox_edges(p)
        box_names[p["title"]] = name
        ib_rows += e
    with open(INTERIM / "infobox_edges.jsonl", "w", encoding="utf-8") as f:
        for r in ib_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Infobox: {len(ib_rows)} cạnh")

    # --- 2b. Tách mục + phân loại trang được link tới ----------------------
    sections = {p["title"]: sections_with_links(p["wikitext"]) for p in tqdm(pages, desc="sections")}
    link_types = {t: "Scientist" for t in scientist_titles}
    cache = INTERIM / "link_types.json"
    alias_path = INTERIM / "aliases.json"
    aliases = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    if cache.exists():
        link_types.update(json.loads(cache.read_text(encoding="utf-8")))
    unknown = {l.title for secs in sections.values() for s in secs for l in s.links}
    unknown |= {r["source"] if r["target"] == r["page"] else r["target"]
                for r in ib_rows if r["is_link"]}
    unknown -= set(link_types)
    if unknown and not args.offline:
        print(f"Phân loại {len(unknown)} trang được liên kết (qua Infobox template)...")
        wc = WikiClient()
        res = classify_titles(wc, list(unknown), SCIENTIST_INFOBOXES, INSTITUTION_INFOBOXES)
        for t, (canon, kind) in res.items():
            link_types[t] = kind
            link_types[canon] = kind
            aliases[t] = canon
    (INTERIM / "aliases.json").write_text(json.dumps(aliases, ensure_ascii=False), encoding="utf-8")
    cache.write_text(json.dumps(link_types, ensure_ascii=False), encoding="utf-8")
    # mở rộng: tên gốc (chưa giải redirect) cũng tra được loại
    for t, c in aliases.items():
        if c in link_types and t not in link_types:
            link_types[t] = link_types[c]

    # --- 2c. NLP ------------------------------------------------------------
    rx = (RelationExtractor if args.legacy else RoleExtractor)(model=args.model, nli_model=args.nli)
    nlp_rows = []
    for p in tqdm(pages, desc="nlp"):
        rels = rx.extract(p["title"], sections[p["title"]], link_types)
        if args.nli:
            rels = rx.verify_with_nli(rels)
        nlp_rows += to_dicts(rels)
    with open(INTERIM / "nlp_edges.jsonl", "w", encoding="utf-8") as f:
        for r in nlp_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"NLP: {len(nlp_rows)} cạnh")


if __name__ == "__main__":
    main()
