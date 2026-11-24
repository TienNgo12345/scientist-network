@echo off
chcp 65001 >nul
REM Chạy đánh giá hỏi đáp với mô hình ngôn ngữ nhỏ (Qwen2.5-7B qua Ollama) trên Windows.
REM Yêu cầu: đã cài Python và Ollama (https://ollama.com). Nháy đúp file này hoặc chạy trong cmd.
cd /d "%~dp0"
echo [1/4] Cai thu vien Python...
python -m pip install -q -r requirements.txt || goto :err
echo [2/4] Tai mo hinh qwen2.5:7b (khoang 4.7 GB, chi lan dau)...
ollama pull qwen2.5:7b || goto :err
echo [3/4] Giai nen du lieu...
python scripts\00_unpack_data.py || goto :err
echo [4/4] Danh gia 144 cau hoi: rules, llm, closed_book (co the mat 30-90 phut)...
set LLM_PROVIDER=ollama
set LLM_MODEL=qwen2.5:7b
set PYTHONIOENCODING=utf-8
python scripts\15_eval_qa.py --eval rules,llm,closed_book || goto :err
echo.
echo XONG. Gui file reports\qa_eval.json cho Claude de dien vao bao cao va slide.
pause
exit /b 0
:err
echo Co loi o buoc tren. Chup man hinh loi va gui lai.
pause
exit /b 1
