"""(Tuỳ chọn) Nhập dữ liệu thu thập bằng trình duyệt thay cho bước 01.

Khi máy chạy pipeline không gọi được API Wikipedia trực tiếp, có thể chạy bộ thu
thập JavaScript (scripts/browser_crawler.js) trong một tab en.wikipedia.org, tải
file JSON (hoặc .json.gz) về rồi nhập bằng script này:

    python scripts/00_import_browser_dump.py scinet_dump.json.gz

Script ghi ra đúng các file mà bước 02 cần: data/raw/scientists.jsonl,
data/interim/link_types.json, data/interim/aliases.json. Sau đó chạy:

    python scripts/02_extract.py --offline
"""
import gzip
import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scinet.config import RAW, INTERIM


def main(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        d = json.load(f)
    pages = d['pages']
    # Loại trang danh sách / trang không có Infobox (không phải tiểu sử nhà khoa học)
    before = len(pages)
    pages = [p for p in pages if not p['title'].startswith(('List of', 'Lists of'))
             and re.search(r'\{\{\s*infobox', p['wikitext'], re.I)]
    print(f"Bỏ {before - len(pages)} trang không phải tiểu sử (danh sách / không có Infobox)")
    with open(RAW / 'scientists.jsonl', 'w', encoding='utf-8') as f:
        for p in pages:
            f.write(json.dumps({'title': p['title'], 'pageid': p['pageid'],
                                'wikitext': p['wikitext'], 'depth': p['depth']},
                               ensure_ascii=False) + '\n')
    (INTERIM / 'link_types.json').write_text(json.dumps(d['link_types'], ensure_ascii=False), encoding='utf-8')
    (INTERIM / 'aliases.json').write_text(json.dumps(d['aliases'], ensure_ascii=False), encoding='utf-8')
    print(f"{len(pages)} trang, {len(d['link_types'])} loại liên kết, {len(d['aliases'])} alias")


if __name__ == '__main__':
    main(sys.argv[1])
