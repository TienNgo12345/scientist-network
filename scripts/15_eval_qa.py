"""Bước 15 – Bộ câu hỏi chuẩn (benchmark) cho hỏi đáp nhiều bước + đánh giá các cấu hình.

Sinh câu hỏi tự động từ đồ thị (có đáp án chuẩn), 6 loại:
  1hop-advisor, 1hop-award, 2hop-advisor-of-advisor, 2hop-students-award(filter),
  2hop-awardees-studied(top), 3hop-advisor-chain
mỗi loại 2 kiểu diễn đạt: "chuẩn" (gần câu mẫu) và "diễn đạt lại" (từ ngữ khác) -> đo độ bền.

Cấu hình đánh giá:
  rules        : lập kế hoạch theo luật + duyệt đồ thị (không LLM)
  llm          : LLM ≤9B lập kế hoạch JSON + duyệt đồ thị (GraphRAG)       [cần LLM]
  closed_book  : LLM trả lời trực tiếp, không có đồ thị (baseline)          [cần LLM]
Chỉ số: Hits@1 (đáp án đầu tiên đúng), Recall@10, F1 tập đáp án, Accuracy kế hoạch hợp lệ.

Chạy: python scripts/15_eval_qa.py --make          (sinh data/final/qa_benchmark.json)
      python scripts/15_eval_qa.py --eval rules     (hoặc llm, closed_book; nhiều cấu hình cách nhau dấu phẩy)
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.graphrag import GraphRAG
from app.kg import FINAL, KnowledgeGraph, strip_accents
from app.llm import LLM

ROOT = Path(__file__).resolve().parent.parent
BENCH = FINAL / "qa_benchmark.json"
ADV = ("DOCTORAL_ADVISOR_OF", "ACADEMIC_ADVISOR_OF")

T = {
    "1hop-advisor": ["Ai là thầy hướng dẫn của {a}?", "Người đã dìu dắt {a} trên con đường nghiên cứu là ai?"],
    "1hop-award": ["{a} đã nhận những giải thưởng nào?", "Những vinh danh mà {a} giành được là gì?"],
    "2hop-advisor-of-advisor": ["Ai là thầy hướng dẫn của thầy hướng dẫn của {a}?",
                                "Ông/bà thầy của người thầy đã đào tạo {a} là ai?"],
    "2hop-students-award": ["Học trò nào của {a} đã nhận {b}?", "Trong số những người {a} từng đào tạo, ai được trao {b}?"],
    "2hop-awardees-studied": ["Những người nhận {b} từng học ở đâu nhiều nhất?",
                              "Các chủ nhân của {b} chủ yếu được đào tạo tại trường nào?"],
    "3hop-advisor-chain": ["Ai là thầy hướng dẫn của thầy hướng dẫn của thầy hướng dẫn của {a}?",
                           "Truy ngược ba thế hệ thầy–trò từ {a} thì gặp ai?"],
}


def ups(kg, n):
    return {h.source for r in ADV for h in kg.neighbors(n, r, "in")}


def downs(kg, n):
    return {h.target for r in ADV for h in kg.neighbors(n, r, "out")}


def make(kg, per_type=12, seed=7):
    rnd = random.Random(seed)
    sci = [n for n, r in kg.nodes.items() if r["label"] == "Scientist"]
    pr = {n: float(kg.metrics.get(n, {}).get("pagerank", 0) or 0) for n in sci}
    famous = sorted(sci, key=lambda n: -pr[n])[:2500]
    qs = []

    def add(t, style, text, gold, kind="set", **kw):
        qs.append({"id": len(qs), "type": t, "style": style, "question": text, "gold": sorted(gold), "kind": kind, **kw})

    pool = [n for n in famous if ups(kg, n)]
    for n in rnd.sample(pool, per_type):
        for s, tpl in enumerate(T["1hop-advisor"]):
            add("1hop-advisor", ["chuẩn", "diễn đạt lại"][s], tpl.format(a=n), ups(kg, n))
    pool = [n for n in famous if 2 <= len(kg.neighbors(n, "RECEIVED_AWARD", "out")) <= 15]
    for n in rnd.sample(pool, per_type):
        g = {h.target for h in kg.neighbors(n, "RECEIVED_AWARD", "out")}
        for s, tpl in enumerate(T["1hop-award"]):
            add("1hop-award", ["chuẩn", "diễn đạt lại"][s], tpl.format(a=n), g)
    pool = [n for n in famous if any(ups(kg, u) for u in ups(kg, n))]
    for n in rnd.sample(pool, per_type):
        g = set().union(*[ups(kg, u) for u in ups(kg, n)]) - {n}
        for s, tpl in enumerate(T["2hop-advisor-of-advisor"]):
            add("2hop-advisor-of-advisor", ["chuẩn", "diễn đạt lại"][s], tpl.format(a=n), g)
    cand = []
    for n in famous:
        aw = defaultdict(set)
        for st in downs(kg, n):
            for h in kg.neighbors(st, "RECEIVED_AWARD", "out"):
                aw[h.target].add(st)
        for a, ss in aw.items():
            if len(ss) >= 1 and kg.U.degree(a) >= 15 and len(a) > 8:
                cand.append((n, a, ss))
    for n, a, ss in rnd.sample(cand, min(per_type, len(cand))):
        for s, tpl in enumerate(T["2hop-students-award"]):
            add("2hop-students-award", ["chuẩn", "diễn đạt lại"][s], tpl.format(a=n, b=a), ss)
    awards = [n for n, r in kg.nodes.items() if r["label"] == "Award" and len(kg.neighbors(n, "RECEIVED_AWARD", "in")) >= 8]
    for a in rnd.sample(awards, min(per_type, len(awards))):
        cnt = defaultdict(int)
        for h in kg.neighbors(a, "RECEIVED_AWARD", "in"):
            for h2 in kg.neighbors(h.source, "STUDIED_AT", "out"):
                cnt[h2.target] += 1
        if not cnt:
            continue
        mx = max(cnt.values())
        gold = {k for k, v in cnt.items() if v == mx}
        for s, tpl in enumerate(T["2hop-awardees-studied"]):
            add("2hop-awardees-studied", ["chuẩn", "diễn đạt lại"][s], tpl.format(b=a), gold, kind="top")
    pool = [n for n in famous if any(ups(kg, u2) for u in ups(kg, n) for u2 in ups(kg, u))]
    for n in rnd.sample(pool, min(per_type, len(pool))):
        g = set()
        for u in ups(kg, n):
            for u2 in ups(kg, u):
                g |= ups(kg, u2)
        g -= {n}
        if not g:
            continue
        for s, tpl in enumerate(T["3hop-advisor-chain"]):
            add("3hop-advisor-chain", ["chuẩn", "diễn đạt lại"][s], tpl.format(a=n), g)
    json.dump(qs, open(BENCH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(qs)} câu hỏi -> {BENCH}")
    return qs


def score(pred, gold, kind):
    g = set(gold)
    if not pred:
        return {"hits1": 0.0, "recall10": 0.0, "f1": 0.0}
    hits1 = float(pred[0] in g)
    rec = len(set(pred[:10]) & g) / len(g) if g else 0.0
    if kind == "top":
        return {"hits1": hits1, "recall10": rec, "f1": hits1}
    ps = set(pred)
    p = len(ps & g) / len(ps); r = len(ps & g) / len(g)
    return {"hits1": hits1, "recall10": rec, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def closed_book(llm, q):
    txt = llm.chat([{"role": "system", "content": "Trả lời câu hỏi về các nhà khoa học. Chỉ liệt kê tên riêng "
                     "(tiếng Anh như trên Wikipedia), mỗi dòng một tên, không giải thích."},
                    {"role": "user", "content": q}], max_tokens=200)
    return [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in txt.splitlines() if l.strip()]


def evaluate(kg, qs, configs):
    llm = LLM()
    if any(c != "rules" for c in configs) and not llm.check():
        print("Không kết nối được LLM:", llm.error)
        configs = [c for c in configs if c == "rules"]
    rag = GraphRAG(kg, llm)
    results = {}
    for cfg in configs:
        rows, t0 = [], time.time()
        for q in qs:
            if cfg == "closed_book":
                pred = closed_book(llm, q["question"])
                pred = [kg.link(p)[0].node if kg.link(p) else p for p in pred]
                plan_ok = None
            else:
                plan = rag.llm_plan(q["question"], kg.link(q["question"])) if cfg == "llm" else None
                if plan is None:
                    plan = rag.rule_plan(q["question"], kg.link(q["question"]))
                items, _ = rag.execute(plan) if plan.intent not in ("none",) else ([], [])
                pred = [i["node"] for i in items]
                plan_ok = plan.intent not in ("none", "global")
            s = score(pred, q["gold"], q["kind"])
            rows.append({**s, "type": q["type"], "style": q["style"], "plan_ok": plan_ok})
        agg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for k in ("hits1", "recall10", "f1"):
                agg["ALL"][k].append(r[k]); agg[r["type"]][k].append(r[k]); agg["style:" + r["style"]][k].append(r[k])
        results[cfg] = {grp: {k: round(sum(v) / len(v), 3) for k, v in d.items()} for grp, d in agg.items()}
        results[cfg]["seconds_per_question"] = round((time.time() - t0) / len(qs), 3)
        if cfg == "llm":
            results[cfg]["model"] = llm.model
        print(cfg, json.dumps(results[cfg]["ALL"]), results[cfg]["seconds_per_question"], "s/câu")
    out = ROOT / "reports" / "qa_eval.json"
    old = json.load(open(out)) if out.exists() else {}
    old.update(results)
    json.dump(old, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--eval", default=None)
    args = ap.parse_args()
    kg = KnowledgeGraph()
    qs = make(kg) if args.make or not BENCH.exists() else json.load(open(BENCH, encoding="utf-8"))
    if args.eval:
        evaluate(kg, qs, args.eval.split(","))


if __name__ == "__main__":
    main()
