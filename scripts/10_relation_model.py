"""Bước 10 – Mô hình nhận dạng quan hệ (giám sát xa) + làm giàu đồ thị.

1. Sinh cặp ứng viên (chủ thể trang, mention đã liên kết) trong từng câu của kho văn bản.
2. Gán nhãn giám sát xa từ các cạnh đã có trong đồ thị.
3. Huấn luyện One-vs-Rest Logistic Regression trên 80% trang, đánh giá trên 20% trang còn lại
   (chia theo trang, cùng hạt giống với bước 9 -> không rò rỉ).
4. Với mỗi loại quan hệ chọn ngưỡng sao cho precision trên tập kiểm tra >= --target-precision.
5. Huấn luyện lại trên toàn bộ, dự đoán trên các cặp CHƯA có cạnh -> cạnh mới (làm giàu).

Đầu ra:
  reports/re_eval.json                 kết quả đánh giá từng quan hệ
  data/enrich/re_new_edges.csv         cạnh mới (source,target,type,prob,evidence,page)
Chạy : python scripts/10_relation_model.py [--target-precision 0.9]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from spacy.tokens import DocBin
from tqdm import tqdm

from scinet.config import DATA, ROOT
from scinet.re_model import LABELS, edge_labels, pair_features, subject_token, syntactic_check, to_edge

ENRICH = DATA / "enrich"
OUT = DATA / "output"
REPORTS = ROOT / "reports"


def build_pairs():
    labels = {r["name"]: r["label"] for r in csv.DictReader(open(OUT / "nodes.csv", encoding="utf-8"))}
    out_edges = defaultdict(set)
    for r in csv.DictReader(open(OUT / "edges.csv", encoding="utf-8")):
        out_edges[(r["source"], r["target"])].add(r["type"])
    meta = [json.loads(l) for l in open(ENRICH / "sections.jsonl", encoding="utf-8")]
    ments = [json.loads(l) for l in open(ENRICH / "mentions.jsonl", encoding="utf-8")]
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    docs = DocBin().from_disk(ENRICH / "docs.spacy").get_docs(nlp.vocab)
    pairs = []
    for m, mm, doc in tqdm(zip(meta, ments, docs), total=len(meta), desc="pairs"):
        page = m["page"]
        base = page.split("(")[0].strip()
        surname = base.split()[-1].lower() if base else ""
        self_spans = [(x["start"], x["end"]) for x in mm["mentions"] if x["kb"] == page]
        objs = [x for x in mm["mentions"] if x["kb"] and x["kb"] != page and x["label"] in
                ("Scientist", "Institution", "Award", "Field")]
        for sent in doc.sents:
            in_sent = [x for x in objs if sent.start_char <= x["start"] < sent.end_char]
            if not in_sent:
                continue
            st = subject_token(sent, self_spans, surname,
                               [(x["start"], x["end"]) for x in mm["mentions"] if x["kb"] and x["kb"] != page])
            if st is None:
                continue
            seen = set()
            for x in in_sent:
                if x["kb"] in seen:
                    continue
                seen.add(x["kb"])
                sp = doc.char_span(x["start"], x["end"], alignment_mode="expand")
                if sp is None or sp.start <= st.i < sp.end:
                    continue
                pairs.append({
                    "page": page, "obj": x["kb"], "olab": x["label"], "source": x["source"],
                    "feats": pair_features(doc, sent, st, sp, x["label"]),
                    "labels": edge_labels(page, x["kb"], out_edges),
                    "has_edge": bool(out_edges.get((page, x["kb"])) or out_edges.get((x["kb"], page))),
                    "syn_ok": syntactic_check(sent, st, sp) and not x["kb"].startswith("List of"),
                    "sent": sent.text.strip(),
                })
    return pairs


def fit(X, y):
    clf = LogisticRegression(C=0.3, max_iter=3000)
    clf.fit(X, y)
    return clf


def pr(y, yhat):
    tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4), tp + fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-precision", type=float, default=0.9)
    args = ap.parse_args()
    pairs = build_pairs()
    print(f"{len(pairs)} cặp ứng viên, {sum(bool(p['labels']) for p in pairs)} có nhãn dương")

    pages = sorted({p["page"] for p in pairs})
    random.Random(42).shuffle(pages)
    train_pages = set(pages[: int(0.8 * len(pages))])
    vec = DictVectorizer()
    X_all = vec.fit_transform([p["feats"] for p in pairs])
    is_train = np.array([p["page"] in train_pages for p in pairs])
    olab = np.array([p["olab"] for p in pairs])

    report = {"pairs": len(pairs), "positive_pairs": sum(bool(p["labels"]) for p in pairs),
              "train_pairs": int(is_train.sum()), "test_pairs": int((~is_train).sum()), "labels": {}}
    thresholds = {}
    for lab, otype in LABELS.items():
        mask = olab == otype
        y = np.array([lab in p["labels"] for p in pairs], dtype=int)
        tr, te = mask & is_train, mask & ~is_train
        if y[tr].sum() < 15 or y[te].sum() < 5:
            report["labels"][lab] = {"skipped": "quá ít mẫu", "pos_train": int(y[tr].sum()),
                                     "pos_test": int(y[te].sum())}
            continue
        clf = fit(X_all[tr], y[tr])
        prob = clf.predict_proba(X_all[te])[:, 1]
        base_p = round(y[te].mean(), 4)                       # baseline: chỉ dựa vào loại đối tượng
        p5 = pr(y[te], (prob >= 0.5).astype(int))
        thr = None
        for t in np.arange(0.5, 0.99, 0.01):
            p, r, f, n = pr(y[te], (prob >= t).astype(int))
            if n >= 10 and p >= args.target_precision:
                thr = round(float(t), 2); pt = (p, r, f, n); break
        report["labels"][lab] = {
            "object_type": otype, "pos_train": int(y[tr].sum()), "pos_test": int(y[te].sum()),
            "neg_test": int((y[te] == 0).sum()),
            "baseline_type_only_precision": base_p,
            "at_0.5": dict(zip(["precision", "recall", "f1", "predicted"], p5)),
            "threshold": thr,
            "at_threshold": dict(zip(["precision", "recall", "f1", "predicted"], pt)) if thr else None,
        }
        if thr:
            thresholds[lab] = thr
        print(lab, report["labels"][lab])

    # ---- Huấn luyện lại trên toàn bộ + dự đoán cạnh mới ------------------------------------
    new = {}
    for lab, thr in thresholds.items():
        otype = LABELS[lab]
        mask = olab == otype
        y = np.array([lab in p["labels"] for p in pairs], dtype=int)
        clf = fit(X_all[mask], y[mask])
        idx = np.where(mask)[0]
        cand = [i for i in idx if not pairs[i]["has_edge"] and pairs[i]["syn_ok"]]
        if not cand:
            continue
        prob = clf.predict_proba(X_all[cand])[:, 1]
        for i, pb in zip(cand, prob):
            if pb < thr:
                continue
            s, t, r = to_edge(pairs[i]["page"], pairs[i]["obj"], lab)
            key = (s, t, r)
            if key not in new or pb > new[key]["prob"]:
                new[key] = {"source": s, "target": t, "type": r, "prob": round(float(pb), 4),
                            "evidence": pairs[i]["sent"], "page": pairs[i]["page"],
                            "mention_source": pairs[i]["source"]}
    # Mỗi cặp chỉ giữ 1 loại quan hệ (xác suất cao nhất), tránh vừa STUDIED_AT vừa WORKED_AT
    best = {}
    for e in new.values():
        k = tuple(sorted([e["source"], e["target"]]))
        if k not in best or e["prob"] > best[k]["prob"]:
            best[k] = e
    rows = sorted(best.values(), key=lambda e: -e["prob"])
    with open(ENRICH / "re_new_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "type", "prob", "evidence", "page",
                                          "mention_source"])
        w.writeheader(); w.writerows(rows)
    report["new_edges"] = dict(Counter(e["type"] for e in rows))
    report["new_edges_total"] = len(rows)
    report["new_edges_from_ner_mentions"] = sum(e["mention_source"] != "wikilink" for e in rows)
    REPORTS.mkdir(exist_ok=True)
    json.dump(report, open(REPORTS / "re_eval.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("new_edges", "new_edges_total",
                                             "new_edges_from_ner_mentions")}, indent=2))


if __name__ == "__main__":
    main()
