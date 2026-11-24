# Mạng lưới các nhà khoa học quốc tế – đồ thị tri thức + GraphRAG
*Đề tài 1 – Phân tích phương tiện mạng xã hội (INT3235E 1), Nhóm 1*

Xây dựng đồ thị tri thức đa quan hệ về các nhà khoa học quốc tế từ **Wikipedia tiếng Anh**
(Infobox + trích xuất quan hệ từ văn bản), làm giàu bằng **NER + nhận dạng quan hệ**, phân tích bằng
các thuật toán mạng xã hội và khai thác bằng ứng dụng **GraphRAG hỏi đáp nhiều bước** chạy với
**mô hình ngôn ngữ nhỏ ≤ 9 tỷ tham số** (mặc định Qwen2.5-7B qua Ollama).

## Kết quả cuối kỳ
| Thành phần | Kết quả |
|---|---|
| Đồ thị cuối (`data/final/`) | **9.271 nút, 27.879 cạnh, 13/13 loại quan hệ có dữ liệu** |
| NER + liên kết thực thể (lai spaCy + gazetteer) | F1 = 0,90 trên 499 trang kiểm tra; +36 nghìn mention ngoài wikilink |
| Nhận dạng quan hệ | luật vai trò + NER: precision 93% (55/59); giám sát xa LogReg: 92% (34/37) – chấm sơ bộ |
| Chất lượng | 33 nút rác bị loại · 0 chu trình thầy–trò · 99,3% cặp thầy–trò hợp lý về năm sinh |
| Thế giới nhỏ | L = 4,50 (ngẫu nhiên 5,39) · 99,5% cặp ≤ 6 bước · σ ≈ 54 |
| Cộng đồng / hạt nhân | Louvain 47 cộng đồng, Q = 0,554 · k-core lớn nhất k = 8, 465 nút |
| Hỏi đáp (144 câu, không LLM) | câu chuẩn Hits@1 = 1,00 · câu diễn đạt lại 0,35 → động lực dùng LLM lập kế hoạch |

## Chạy ứng dụng (5 phút)
```bash
pip install -r requirements.txt
cp .env.example .env              # chọn nhà cung cấp LLM; LLM_PROVIDER=none nếu chưa có
# (tuỳ chọn) cài Ollama rồi: ollama pull qwen2.5:7b
streamlit run app/streamlit_app.py
```
Ba trang: **Hỏi đáp GraphRAG** (chuỗi suy luận + đồ thị bằng chứng), **Chân dung nhà khoa học**
(số đo mạng, mạng lân cận, phả hệ, cộng đồng, nhà khoa học tương tự) và **Tổng quan mạng**.
Kiểm thử: `python tests/test_app.py` → `OK`.

## Pipeline cuối kỳ (sau các bước 0–7 của giữa kỳ)
```bash
python scripts/08_enrich_corpus.py        # kho văn bản làm giàu: 4.274 mục, 29.854 câu (spaCy)
python scripts/09_ner_link.py             # NER + liên kết thực thể, có đánh giá -> reports/ner_eval.json
python scripts/10_relation_model.py       # RE giám sát xa (LogReg) -> data/enrich/re_new_edges.csv
python scripts/11_enrich_rules.py         # RE theo vai trò ngữ pháp trên mention NER
python scripts/12_quality_and_merge.py    # đánh giá chất lượng + đồ thị cuối -> data/final/
python scripts/13_sna_algorithms.py       # thế giới nhỏ, PageRank, Louvain, k-core -> reports/sna.json
python scripts/14_community_summaries.py  # tóm tắt cộng đồng cho GraphRAG (--llm để LLM viết)
python scripts/15_eval_qa.py --eval rules,llm,closed_book   # đánh giá hỏi đáp nhiều bước
```
Các thuật toán tương ứng trên Neo4j GDS nằm ở cuối `queries.cypher`.

---
## Phần giữa kỳ
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

## Kết quả giữa kỳ (thu thập ngày 07/11/2026)
> Ghi chú cuối kỳ: đã sửa lỗi 3 trang giải thưởng (Fields Medal, Kyoto Prize…) bị coi là tiểu sử ->
> `data/output` hiện có 9.306 nút, 27.840 cạnh (+77 cạnh "nhận Huy chương Fields/Kyoto").
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
