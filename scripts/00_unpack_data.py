"""Giải nén dữ liệu gốc đã nén trong repo (GitHub giới hạn kích thước file).

    python scripts/00_unpack_data.py

Tạo lại:
  data/raw/scientists.jsonl        – wikitext 3.141 trang đã crawl (đầu vào của bước 02 và 08)
  data/enrich/sections.jsonl       – kho văn bản làm giàu: 4.274 mục, 29.854 câu
  data/enrich/mentions.jsonl       – đề cập thực thể do mô hình NER + liên kết tìm được (đầu ra bước 09)

File docs.spacy (bản phân tích spaCy) và data/interim/ không đưa lên vì sinh lại được:
  python scripts/02_extract.py && python scripts/08_enrich_corpus.py
"""
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

n = 0
for gz in sorted((ROOT / "data").rglob("*.gz")):
    out = gz.with_suffix("")
    if out.exists() and out.stat().st_mtime >= gz.stat().st_mtime:
        print(f"đã có  {out.relative_to(ROOT)}")
        continue
    with gzip.open(gz, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    n += 1
    print(f"giải nén {gz.relative_to(ROOT)} -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")
print(f"Xong ({n} file mới).")
