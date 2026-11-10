# Mạng lưới các nhà khoa học quốc tế (Đề tài 1 – Phân tích phương tiện mạng xã hội)

Xây dựng đồ thị tri thức đa quan hệ về các nhà khoa học quốc tế từ **Wikipedia tiếng Anh**,
kết hợp **Infobox** (bán cấu trúc) và **trích xuất quan hệ bằng NLP** trên các mục
"Scientific career", "Research", "Controversies"..., lưu trữ trong **Neo4j**.

- 4 loại nút: `Scientist`, `Institution`, `Field`, `Award`
- 13 loại quan hệ:
  - Infobox: `DOCTORAL_ADVISOR_OF`, `ACADEMIC_ADVISOR_OF`, `INFLUENCED`, `WORKED_AT`,
    `STUDIED_AT`, `WORKS_IN_FIELD`, `RECEIVED_AWARD`
  - NLP: `BUILT_UPON` (kế thừa học thuật), `CRITICIZED` (phản biện), `REFUTED` (bác bỏ),
    `PRIORITY_DISPUTE` (tranh chấp ưu tiên), `COLLABORATED_WITH` (hợp tác),
    `ADVISOR_TO` (cố vấn tại viện/tổ chức quốc tế)

## Cài đặt
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm                # hoặc en_core_web_trf (chính xác hơn)
docker compose up -d                                   # Neo4j 5 + GDS tại http://localhost:7474
```

## Kết quả hiện có (thu thập ngày 07/11/2026)
- 3.141 trang tiểu sử → **9.303 nút, 27.763 cạnh, 12 loại quan hệ có dữ liệu**
- `data/output/nodes.csv`, `data/output/edges.csv` – mạng hoàn chỉnh (mỗi cạnh có câu bằng chứng)
- `reports/stats.json`, `stats.md`, `*.png` – thống kê và hình
- `reports/preliminary_precision.md` – chấm sơ bộ độ chính xác (≈ 98,8% trên 336 cạnh mẫu)
- `data/output/eval_sample.csv` – 511 cạnh mẫu để chấm chính thức

## Chạy pipeline
```bash
python scripts/01_crawl.py --target 3200       # thu thập (~15–30 phút)
#   Nếu máy không gọi được API Wikipedia: mở https://en.wikipedia.org/wiki/Special:BlankPage,
#   dán scripts/browser_crawler.js vào Console, đợi S.done === true, gọi `await exportDump()`,
#   rồi: python scripts/00_import_browser_dump.py scinet_dump.json.gz
python scripts/02_extract.py                   # Infobox + NLP (thêm --offline nếu dùng dữ liệu từ trình duyệt;
                                               #  --nli facebook/bart-large-mnli để kiểm chứng; --legacy = bộ trích xuất v1)
python scripts/03_build_graph.py               # hợp nhất -> data/output/nodes.csv, edges.csv
python scripts/04_load_neo4j.py                # nạp vào Neo4j
python scripts/05_network_stats.py             # thống kê -> reports/stats.json, *.png
python scripts/06_eval_sample.py sample        # lấy mẫu để chấm tay độ chính xác
python scripts/06_eval_sample.py score         # tính precision + khoảng tin cậy 95%
python scripts/07_figures.py                   # vẽ hình cho báo cáo
python tests/test_pipeline.py                  # kiểm thử offline
```
Truy vấn mẫu và thuật toán GDS (PageRank, Betweenness, Louvain): `queries.cypher`.

## Cấu trúc
```
scinet/config.py              cấu hình: seed category, ánh xạ Infobox -> quan hệ
scinet/wiki_client.py         MediaWiki API: category, wikitext theo lô, redirect, phân loại trang
scinet/wikitext.py            parse Infobox, làm sạch văn bản, giữ vị trí wikilink
scinet/relation_extractor.py  tách câu, POS, dependency, đồng tham chiếu, trigger, NLI
scripts/01..06                các bước pipeline
```
