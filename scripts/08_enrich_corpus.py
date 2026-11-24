"""Bước 8 – Kho văn bản làm giàu (enrichment corpus).

Lấy TOÀN BỘ các mục văn bản (không chỉ Career/Controversies) của 3.141 trang tiểu sử,
tách câu + phân tích cú pháp + NER bằng spaCy một lần rồi lưu lại để các bước 9–10 dùng chung.

Đầu vào : data/raw/scientists.jsonl
Đầu ra  : data/enrich/docs.spacy        (DocBin: token, lemma, POS, dependency, NER)
          data/enrich/sections.jsonl    (page, heading, text, links[start,end,title,surface])
          data/enrich/corpus_stats.json
Chạy    : python scripts/08_enrich_corpus.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

from scinet.config import RAW, DATA
from scinet.wiki_client import read_jsonl
from scinet.wikitext import sections_with_links

ENRICH = DATA / "enrich"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()
    ENRICH.mkdir(parents=True, exist_ok=True)

    pages = list(read_jsonl(RAW / "scientists.jsonl"))[: args.limit]
    meta = []
    for p in tqdm(pages, desc="sections"):
        for s in sections_with_links(p["wikitext"], only_relevant=False):
            if len(s.text) < 40:
                continue
            meta.append({"page": p["title"], "heading": s.heading, "text": s.text,
                         "links": [[l.start, l.end, l.title, l.surface] for l in s.links]})

    nlp = spacy.load(args.model)
    nlp.max_length = 2_000_000
    db = DocBin(attrs=["ORTH", "LEMMA", "POS", "TAG", "DEP", "HEAD", "ENT_IOB", "ENT_TYPE",
                       "SENT_START", "SPACY"])
    n_sent = n_tok = 0
    for doc in tqdm(nlp.pipe((m["text"] for m in meta), batch_size=64), total=len(meta), desc="spacy"):
        db.add(doc)
        n_tok += len(doc)
        n_sent += sum(1 for _ in doc.sents)
    db.to_disk(ENRICH / "docs.spacy")
    with open(ENRICH / "sections.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    stats = {"pages": len(pages), "sections": len(meta), "sentences": n_sent, "tokens": n_tok,
             "wikilinks": sum(len(m["links"]) for m in meta)}
    json.dump(stats, open(ENRICH / "corpus_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
