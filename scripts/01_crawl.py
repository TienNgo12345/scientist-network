"""Bước 1 – Thu thập dữ liệu: seed từ các Category + mở rộng BFS theo Infobox/văn bản.

Kết quả: data/raw/scientists.jsonl  (mỗi dòng: title, pageid, wikitext, depth)
Chạy:  python scripts/01_crawl.py --target 2500
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm

from scinet.config import (RAW, SEED_CATEGORIES, TARGET_SCIENTISTS, MAX_BFS_DEPTH,
                           SCIENTIST_INFOBOXES, INSTITUTION_INFOBOXES, INFOBOX_FIELD_MAP)
from scinet.wiki_client import WikiClient, append_jsonl, read_jsonl, classify_titles
from scinet.wikitext import parse_infobox, sections_with_links

OUT = RAW / "scientists.jsonl"
PERSON_FIELDS = [k for k, v in INFOBOX_FIELD_MAP.items() if v[2] == "Scientist"]


def neighbours(wikitext: str):
    """Láng giềng ưu tiên: người trong Infobox (thầy/trò/ảnh hưởng) + link trong mục sự nghiệp."""
    _, box = parse_infobox(wikitext)
    first = [t for f in PERSON_FIELDS for t in box.get(f, {}).get("links", [])]
    second = [l.title for s in sections_with_links(wikitext) for l in s.links]
    return list(dict.fromkeys(first + second))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_SCIENTISTS)
    ap.add_argument("--per-category", type=int, default=400)
    args = ap.parse_args()

    wc = WikiClient()
    done = {r["title"] for r in read_jsonl(OUT)} if OUT.exists() else set()
    print(f"Đã có {len(done)} trang trong cache")

    # 1. Seeds
    seeds = []
    for cat in SEED_CATEGORIES:
        mem = wc.category_members(cat, limit=args.per_category)
        print(f"  {cat}: {len(mem)} trang")
        seeds += mem
    seeds = list(dict.fromkeys(seeds))

    queue = deque((t, 0) for t in seeds)
    seen = set(seeds) | done
    pbar = tqdm(total=args.target, initial=len(done), desc="scientists")
    count = len(done)

    while queue and count < args.target:
        batch = [queue.popleft() for _ in range(min(50, len(queue)))]
        depth_of = {t: d for t, d in batch}
        kinds = classify_titles(wc, list(depth_of), SCIENTIST_INFOBOXES, INSTITUTION_INFOBOXES)
        # seed (depth 0) được nhận kể cả khi dùng Infobox person; các tầng sau phải có infobox nhà khoa học
        keep = [t for t, (c, k) in kinds.items() if k == "Scientist" or depth_of[t] == 0]
        pages = wc.fetch_wikitext(keep)
        rows = []
        for t, p in pages.items():
            if p["title"] in done:
                continue
            done.add(p["title"])
            p["depth"] = depth_of[t]
            rows.append(p)
            if depth_of[t] < MAX_BFS_DEPTH:
                for nb in neighbours(p["wikitext"]):
                    if nb not in seen:
                        seen.add(nb)
                        queue.append((nb, depth_of[t] + 1))
        append_jsonl(OUT, rows)
        count += len(rows)
        pbar.update(len(rows))
    pbar.close()
    print(f"Tổng số trang nhà khoa học: {count}. File: {OUT}")


if __name__ == "__main__":
    main()
