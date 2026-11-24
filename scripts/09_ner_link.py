"""Bước 9 – Nhận dạng thực thể (NER) + liên kết thực thể (EL), có đánh giá.

Đánh giá (không rò rỉ nhãn): chia trang 80/20. Bảng bí danh chỉ học từ anchor text của
80% trang huấn luyện; mô hình chạy trên VĂN BẢN THÔ của 20% trang kiểm tra (đã bỏ link),
rồi so với các wikilink trỏ tới nút trong đồ thị (nhãn bạc – silver standard).

Đầu ra:
  data/enrich/mentions.jsonl   mention trên toàn kho (wikilink + mention mới do NER phát hiện)
  reports/ner_eval.json        kết quả đánh giá các cấu hình
  reports/ner_new_entities.csv thực thể PERSON/ORG chưa có trong đồ thị, xếp theo tần suất
Chạy : python scripts/09_ner_link.py
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

from scinet.config import DATA, INTERIM, ROOT
from scinet.ner import AliasTable, EntityRecognizer

ENRICH = DATA / "enrich"
OUT = DATA / "output"
REPORTS = ROOT / "reports"


def load():
    labels = {}
    with open(OUT / "nodes.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels[r["name"]] = r["label"]
    meta = [json.loads(l) for l in open(ENRICH / "sections.jsonl", encoding="utf-8")]
    nlp = spacy.load("en_core_web_sm")
    docs = list(DocBin().from_disk(ENRICH / "docs.spacy").get_docs(nlp.vocab))
    aliases = json.load(open(INTERIM / "aliases.json", encoding="utf-8"))
    return labels, meta, docs, nlp, aliases


def build_alias(labels, aliases, meta, pages):
    al = AliasTable(labels)
    al.add_redirects({a: t for a, t in aliases.items() if t in labels})
    for m in meta:
        if m["page"] in pages:
            for s, e, t, surf in m["links"]:
                t = aliases.get(t, t)
                al.add(surf, t, 1)
    return al


def evaluate(rec, meta, docs, test_pages, labels, aliases):
    gold_n = det_strict = det_relaxed = linked_ok = pred_on_gold = pred_on_gold_ok = 0
    new_mentions = 0
    by_label = defaultdict(lambda: [0, 0])
    for m, doc in zip(meta, docs):
        if m["page"] not in test_pages:
            continue
        gold = []
        for s, e, t, surf in m["links"]:
            t = aliases.get(t, t)
            if t in labels:
                gold.append((s, e, t))
        preds = rec.recognize(doc, page_title=m["page"], page_context=None)
        gold_n += len(gold)
        for s, e, t in gold:
            lab = labels[t]
            by_label[lab][0] += 1
            hit = [p for p in preds if not (p.end <= s or p.start >= e)]
            if hit:
                det_relaxed += 1
                if any(p.start == s and p.end == e for p in hit):
                    det_strict += 1
                if any(p.kb == t for p in hit):
                    linked_ok += 1
                    by_label[lab][1] += 1
        for p in preds:
            ov = [g for g in gold if not (p.end <= g[0] or p.start >= g[1])]
            if ov:
                pred_on_gold += 1
                pred_on_gold_ok += int(p.kb == ov[0][2])
            else:
                new_mentions += 1
    r = lambda a, b: round(a / b, 4) if b else 0.0
    return {
        "gold_mentions": gold_n,
        "detection_recall_relaxed": r(det_relaxed, gold_n),
        "detection_recall_strict": r(det_strict, gold_n),
        "end_to_end_recall": r(linked_ok, gold_n),
        "linking_precision_on_gold_region": r(pred_on_gold_ok, pred_on_gold),
        "f1_end_to_end": r(2 * r(linked_ok, gold_n) * r(pred_on_gold_ok, pred_on_gold),
                           r(linked_ok, gold_n) + r(pred_on_gold_ok, pred_on_gold)),
        "mentions_outside_wikilinks": new_mentions,
        "recall_by_label": {k: r(v[1], v[0]) for k, v in by_label.items()},
    }


def main():
    labels, meta, docs, nlp, aliases = load()
    pages = sorted({m["page"] for m in meta})
    random.Random(42).shuffle(pages)
    cut = int(0.8 * len(pages))
    train, test = set(pages[:cut]), set(pages[cut:])
    print(f"{len(meta)} mục, {len(pages)} trang -> train {len(train)} / test {len(test)}")

    al_train = build_alias(labels, aliases, meta, train)
    configs = {
        "A_spacy_ner_only": dict(use_gazetteer=False, use_spacy=True),
        "B_gazetteer_only": dict(use_gazetteer=True, use_spacy=False),
        "C_hybrid": dict(use_gazetteer=True, use_spacy=True),
    }
    results = {"split": {"train_pages": len(train), "test_pages": len(test)}}
    for name, cfg in configs.items():
        rec = EntityRecognizer(nlp, labels, al_train, use_context=False, **cfg)
        results[name] = evaluate(rec, meta, docs, test, labels, aliases)
        print(name, json.dumps(results[name], ensure_ascii=False))
    REPORTS.mkdir(exist_ok=True)
    json.dump(results, open(REPORTS / "ner_eval.json", "w"), indent=2, ensure_ascii=False)

    # ---- Suy luận trên toàn kho (bảng bí danh từ toàn bộ anchor, có ngữ cảnh trang) ----------
    al_all = build_alias(labels, aliases, meta, set(pages))
    rec = EntityRecognizer(nlp, labels, al_all, use_context=True)
    page_ctx = defaultdict(set)
    for m in meta:
        for s, e, t, surf in m["links"]:
            t = aliases.get(t, t)
            if t in labels:
                page_ctx[m["page"]].add(t)
    new_ent = Counter()
    n_wiki = n_new_linked = n_new_unlinked = 0
    with open(ENRICH / "mentions.jsonl", "w", encoding="utf-8") as f:
        for i, (m, doc) in enumerate(tqdm(list(zip(meta, docs)), desc="NER+EL")):
            gold = []
            for s, e, t, surf in m["links"]:
                t = aliases.get(t, t)
                if t in labels:
                    gold.append({"start": s, "end": e, "text": surf, "label": labels[t], "kb": t,
                                 "score": 1.0, "source": "wikilink"})
            n_wiki += len(gold)
            preds = rec.recognize(doc, m["page"], page_ctx[m["page"]])
            extra = []
            for p in preds:
                if any(not (p.end <= g["start"] or p.start >= g["end"]) for g in gold):
                    continue
                extra.append(p.__dict__)
                if p.kb:
                    n_new_linked += 1
                else:
                    n_new_unlinked += 1
                    if p.label == "PERSON" and len(p.text.split()) >= 2:
                        new_ent[(p.text, p.label)] += 1
            f.write(json.dumps({"i": i, "page": m["page"], "mentions": gold + extra},
                               ensure_ascii=False) + "\n")
    summary = {"wikilink_mentions": n_wiki, "new_linked_mentions": n_new_linked,
               "new_unlinked_mentions": n_new_unlinked,
               "gain_linked_pct": round(100 * n_new_linked / max(n_wiki, 1), 1)}
    results["inference"] = summary
    json.dump(results, open(REPORTS / "ner_eval.json", "w"), indent=2, ensure_ascii=False)
    with open(REPORTS / "ner_new_entities.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["entity", "type", "mentions"])
        for (t, l), c in new_ent.most_common(300):
            w.writerow([t, l, c])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
