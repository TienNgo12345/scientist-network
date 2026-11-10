"""Bước 6 – Đánh giá độ chính xác (precision) bằng gán nhãn thủ công.

  python scripts/06_eval_sample.py sample --per-type 50
      -> data/output/eval_sample.csv (cột 'correct' để trống, người chấm điền 1/0)
  python scripts/06_eval_sample.py score
      -> precision theo từng loại quan hệ + khoảng tin cậy Wilson 95%
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from scinet.config import OUT

PATH = OUT / "eval_sample.csv"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--per-type", type=int, default=50)
    a = ap.parse_args()
    if a.cmd == "sample":
        e = pd.read_csv(OUT / "edges.csv")
        s = (e.groupby("type", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), a.per_type), random_state=42)))
        s["correct"] = ""
        s.to_csv(PATH, index=False)
        print(f"Đã lấy {len(s)} cạnh -> {PATH}. Hãy điền cột 'correct' (1 đúng / 0 sai).")
    else:
        s = pd.read_csv(PATH)
        s = s[s.correct.isin([0, 1])]
        e = pd.read_csv(OUT / "edges.csv")
        w = e.type.value_counts()
        rows = []
        for t, g in s.groupby("type"):
            k, n = int(g.correct.sum()), len(g)
            lo, hi = wilson(k, n)
            rows.append({"type": t, "n": n, "precision": round(k / n, 3),
                         "ci95": f"[{lo:.3f}, {hi:.3f}]", "edges_total": int(w.get(t, 0))})
        r = pd.DataFrame(rows)
        overall = (r.precision * r.edges_total).sum() / r.edges_total.sum()
        print(r.to_string(index=False))
        print(f"Precision tổng (trọng số theo số cạnh): {overall:.3f}")


if __name__ == "__main__":
    main()
