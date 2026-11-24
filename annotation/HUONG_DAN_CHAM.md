# Hướng dẫn chấm cho người chấm thứ hai

Mục đích: đo độ đồng thuận giữa hai người chấm (Cohen's kappa) cho các cạnh mới do mô hình làm giàu tạo ra.
Người chấm thứ hai **không được xem** nhãn của người thứ nhất (cột `correct` trong `data/enrich/audit_new_edges.csv`).

## Cách làm
1. Mở `annotation/audit_annotator2.csv` bằng Excel hoặc Google Sheets (96 dòng).
2. Với mỗi dòng, đọc câu ở cột `evidence` và trả lời: câu này có nói rằng `source` —`type`→ `target` không?
   - Ghi `1` vào cột `correct` nếu ĐÚNG, `0` nếu SAI.
   - Chỉ dựa vào câu bằng chứng; được tra Wikipedia nếu câu mơ hồ về người được nhắc tới.
3. Lưu lại dạng CSV (UTF-8), giữ nguyên tên file và các cột.
4. Chạy `python scripts/16_agreement.py` → in kappa, độ chính xác và danh sách cạnh hai người bất đồng.
5. Hai người thảo luận các cạnh bất đồng; ghi kết quả vào báo cáo (mục Quality assessment).

## Định nghĩa nhãn quan hệ
| Quan hệ | Đúng khi câu nói rằng |
|---|---|
| COLLABORATED_WITH | hai người cùng làm việc / cùng công bố / cùng nghiên cứu |
| CRITICIZED | `source` phê phán, phản đối ý kiến hoặc công trình của `target` |
| REFUTED | `source` chứng minh công trình hoặc giả thuyết của `target` là sai |
| BUILT_UPON | `source` phát triển tiếp trên công trình của `target` |
| ADVISOR_TO | `source` làm cố vấn (không phải hướng dẫn tiến sĩ) cho người hoặc tổ chức `target` |
| PRIORITY_DISPUTE | hai người tranh chấp ai phát hiện trước |
| STUDIED_AT | `source` học tại cơ sở `target` |
| RECEIVED_AWARD | `source` nhận giải `target` |

Mẹo: nếu chiều quan hệ bị ngược (B phê bình A nhưng file ghi A phê bình B) thì chấm `0`.

## Mức kappa (Landis & Koch)
< 0,2 kém · 0,2–0,4 trung bình yếu · 0,4–0,6 trung bình · 0,6–0,8 tốt · > 0,8 rất tốt
