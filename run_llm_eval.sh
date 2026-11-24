#!/usr/bin/env bash
# Chạy đánh giá hỏi đáp với mô hình ngôn ngữ nhỏ (Qwen2.5-7B qua Ollama) trên macOS / Linux.
# Yêu cầu: Python 3.10+ và Ollama (https://ollama.com).   bash run_llm_eval.sh
set -e
cd "$(dirname "$0")"
echo "[1/4] Cài thư viện Python..."; python3 -m pip install -q -r requirements.txt
echo "[2/4] Tải mô hình qwen2.5:7b (~4,7 GB, chỉ lần đầu)..."; ollama pull qwen2.5:7b
echo "[3/4] Giải nén dữ liệu..."; python3 scripts/00_unpack_data.py
echo "[4/4] Đánh giá 144 câu hỏi (30–90 phút)..."
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b python3 scripts/15_eval_qa.py --eval rules,llm,closed_book
echo "XONG. Gửi file reports/qa_eval.json để điền vào báo cáo và slide."
