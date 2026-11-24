"""Độ đồng thuận giữa hai người chấm (Cohen's kappa) và độ chính xác có khoảng tin cậy Wilson.

Người chấm 1 (Ngô Mạnh Tiến) đã chấm cột `correct` trong data/enrich/audit_new_edges.csv.
Người chấm 2 (Dương Việt Hùng) chấm ĐỘC LẬP file annotation/audit_annotator2.csv
(cột `correct`: 1 = đúng, 0 = sai, để trống = chưa chấm) mà KHÔNG xem nhãn của người 1.

    python scripts/16_agreement.py
    python scripts/16_agreement.py --a data/enrich/audit_new_edges.csv --b annotation/audit_annotator2.csv

Kết quả: in ra màn hình và ghi reports/agreement.json.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEY = ["source", "target", "type"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def interpret(k):
    for lim, name in [(0.2, "kém"), (0.4, "trung bình yếu"), (0.6, "trung bình"), (0.8, "tốt"), (1.01, "rất tốt")]:
        if k < lim:
            return name
    return "rất tốt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "data/enrich/audit_new_edges.csv"))
    ap.add_argument("--b", default=str(ROOT / "annotation/audit_annotator2.csv"))
    ap.add_argument("--out", default=str(ROOT / "reports/agreement.json"))
    args = ap.parse_args()

    A = pd.read_csv(args.a)
    B = pd.read_csv(args.b)
    # cạnh trùng khoá (cùng cặp, cùng quan hệ, khác câu bằng chứng) được ghép theo thứ tự xuất hiện
    A["_k"] = A.groupby(KEY).cumcount()
    B["_k"] = B.groupby(KEY).cumcount()
    m = A.merge(B[KEY + ["_k", "correct"]], on=KEY + ["_k"], suffixes=("_a", "_b"))
    m = m.dropna(subset=["correct_a", "correct_b"])
    if m.empty:
        raise SystemExit("Người chấm 2 chưa chấm dòng nào (cột correct còn trống).")
    a = m["correct_a"].astype(int).tolist()
    b = m["correct_b"].astype(int).tolist()
    k, po = kappa(a, b)

    # nhãn hợp nhất: đúng khi cả hai cùng nói đúng (chặt); bất đồng liệt kê ra để thảo luận
    m["agreed"] = ((m["correct_a"] == 1) & (m["correct_b"] == 1)).astype(int)
    res = {"n_items": len(m), "observed_agreement": round(po, 4), "cohen_kappa": round(k, 4),
           "kappa_level": interpret(k), "by_method": {}}
    for meth, g in m.groupby("method"):
        p, lo, hi = wilson(int(g["agreed"].sum()), len(g))
        res["by_method"][meth] = {"n": len(g), "precision_both_correct": round(p, 4),
                                  "ci95": [round(lo, 4), round(hi, 4)]}
    p, lo, hi = wilson(int(m["agreed"].sum()), len(m))
    res["overall"] = {"precision_both_correct": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    dis = m[m["correct_a"] != m["correct_b"]][KEY + ["correct_a", "correct_b", "evidence"]]
    res["disagreements"] = dis.to_dict("records")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Số cạnh chấm chung: {len(m)}")
    print(f"Đồng thuận quan sát: {po:.1%}   Cohen's kappa: {k:.3f} ({interpret(k)})")
    for meth, r in res["by_method"].items():
        print(f"  {meth:15s} n={r['n']:3d}  độ chính xác {r['precision_both_correct']:.1%}  CI95 {r['ci95']}")
    print(f"Bất đồng: {len(dis)} cạnh (xem {args.out})")


if __name__ == "__main__":
    main()
