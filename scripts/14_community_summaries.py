"""Bước 14 – Tóm tắt cộng đồng cho GraphRAG (global search, theo Edge et al. 2024).

Mỗi cộng đồng Louvain được mô tả bằng: kích thước, lĩnh vực / viện / giải nổi bật, nhà khoa học
trung tâm (PageRank), hạt nhân (k-core trong cộng đồng) và các quan hệ học thuật tiêu biểu.
  --llm : dùng mô hình ngôn ngữ nhỏ (cấu hình trong .env) viết lại thành đoạn văn tiếng Việt.
          Không có cờ này -> tóm tắt dạng mẫu câu (chạy được không cần LLM).
Đầu ra: data/final/community_summaries.json
Chạy  : python scripts/14_community_summaries.py [--llm] [--top 40]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.kg import FINAL, KnowledgeGraph
from app.llm import LLM


def facts(kg, c):
    members = set(c["top_scientists_pagerank"]) | set(c["nucleus"])
    rels = []
    for n in list(members)[:12]:
        for h in kg.neighbors(n, None, "out"):
            if h.rel in ("DOCTORAL_ADVISOR_OF", "INFLUENCED", "COLLABORATED_WITH", "CRITICIZED", "BUILT_UPON") \
                    and h.target in members:
                rels.append(kg.describe(h))
    return rels[:10]


def template(kg, c):
    s = (f"Cộng đồng {c['id']} gồm {c['size']} nút ({c['labels'].get('Scientist', 0)} nhà khoa học). "
         f"Lĩnh vực chủ đạo: {', '.join(c['top_fields']) or 'không rõ'}. "
         f"Viện tiêu biểu: {', '.join(c['top_institutions'][:3]) or 'không rõ'}. "
         f"Giải thưởng phổ biến: {', '.join(c['top_awards'][:3]) or 'không rõ'}. "
         f"Nhà khoa học trung tâm (PageRank): {', '.join(c['top_scientists_pagerank'][:5])}. "
         f"Hạt nhân k-core (k={c['k_core_max']}): {', '.join(c['nucleus'][:5])}.")
    f = facts(kg, c)
    if f:
        s += " Quan hệ tiêu biểu: " + "; ".join(f[:5]) + "."
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    kg = KnowledgeGraph()
    llm = LLM() if args.llm else None
    out = {}
    for c in kg.communities[: args.top]:
        base = template(kg, c)
        if llm and llm.available:
            try:
                txt = llm.chat([{"role": "system", "content": "Viết 3-4 câu tiếng Việt tóm tắt một cộng đồng trong mạng "
                                 "các nhà khoa học: chủ đề nghiên cứu, trung tâm, nhân vật nòng cốt và quan hệ nổi bật. "
                                 "Chỉ dùng dữ kiện được cho, giữ tên riêng tiếng Anh."},
                                {"role": "user", "content": base}], max_tokens=300)
                out[str(c["id"])] = txt.strip()
                print(c["id"], "LLM ok")
                continue
            except Exception as e:
                print("LLM lỗi, dùng mẫu câu:", str(e)[:120])
        out[str(c["id"])] = base
    json.dump(out, open(FINAL / "community_summaries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} tóm tắt -> {FINAL / 'community_summaries.json'}")
    print(out.get("0", "")[:500])


if __name__ == "__main__":
    main()
