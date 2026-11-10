# Chấm sơ bộ độ chính xác (giữa kỳ)

Người chấm: Claude (trợ lý AI) đọc câu bằng chứng / tên nút, ngày 07/11/2026.
Đây là chấm sơ bộ để kiểm tra pipeline; nhóm cần chấm lại độc lập theo mục 9.3
(06_eval_sample.py sample/score) cho đánh giá chính thức ở bài cuối kỳ.

| Loại quan hệ | Cách lấy mẫu | Số mẫu | Đúng | Precision |
| --- | --- | --- | --- | --- |
| 7 loại Infobox | 25 cạnh ngẫu nhiên mỗi loại (seed 5) | 175 | 173 | 98,9% |
| CRITICIZED | toàn bộ | 33 | 33 | 100% |
| REFUTED | toàn bộ | 5 | 5 | 100% |
| BUILT_UPON | toàn bộ | 24 | 21 | 87,5% |
| COLLABORATED_WITH | 50 cạnh ngẫu nhiên (seed 11) | 50 | 48 | 96,0% |
| ADVISOR_TO | toàn bộ | 49 | 47 | 95,9% |
| PRIORITY_DISPUTE | không có cạnh trên dữ liệu thật | 0 | – | – |
| **Tổng có trọng số theo số cạnh** | | 336 | 327 | **≈ 98,8%** |

Lỗi gặp phải:
- Infobox: "Boron", "Chemist" bị đưa vào lĩnh vực (fields).
- BUILT_UPON: "inspired him to pursue an academic career" (cảm hứng nghề nghiệp, không phải kế thừa công trình) ×2; một câu về Birkar–Shokurov không đủ căn cứ.
- COLLABORATED_WITH: "had planned to work with" (dự định, chưa làm); một câu "along with Dijkstra" không rõ.
- ADVISOR_TO: cố vấn cho JCESR nhưng link tới Argonne; Janet Currie gán sai tổ chức.
