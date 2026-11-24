"""Client cho mô hình ngôn ngữ nhỏ (≤ 9 tỷ tham số) qua API tương thích OpenAI.

Hỗ trợ (chọn bằng biến môi trường hoặc file .env ở thư mục gốc):
  LLM_PROVIDER=ollama      (mặc định)  LLM_MODEL=qwen2.5:7b        – chạy cục bộ: `ollama pull qwen2.5:7b`
  LLM_PROVIDER=openrouter              LLM_MODEL=meta-llama/llama-3.1-8b-instruct  + OPENROUTER_API_KEY
  LLM_PROVIDER=openai_compatible       LLM_BASE_URL=...  LLM_MODEL=...  LLM_API_KEY=...
  LLM_PROVIDER=none                    – không dùng LLM, ứng dụng chạy bằng bộ lập kế hoạch theo luật

Mọi model đều phải ≤ 9B tham số theo yêu cầu đề bài; bảng MODEL_PARAMS ghi lại kích thước để
giao diện hiển thị và chặn model quá lớn.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent

MODEL_PARAMS = {  # tỷ tham số
    "qwen2.5:7b": 7.6, "qwen2.5:3b": 3.1, "qwen2.5:1.5b": 1.5, "qwen3:8b": 8.2, "qwen3:4b": 4.0,
    "llama3.1:8b": 8.0, "gemma2:9b": 9.2, "gemma3:4b": 4.3, "phi3.5": 3.8, "mistral:7b": 7.2,
    "meta-llama/llama-3.1-8b-instruct": 8.0, "qwen/qwen-2.5-7b-instruct": 7.6,
    "google/gemma-2-9b-it": 9.2, "llama-3.1-8b-instant": 8.0,
}
PROVIDERS = {
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b", None),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.1-8b-instruct", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant", "GROQ_API_KEY"),
}


def _load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


class LLM:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        _load_env()
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.client = None
        self.model = model or os.getenv("LLM_MODEL")
        self.error = None
        if self.provider == "none":
            return
        base, default_model, key_env = PROVIDERS.get(self.provider, (os.getenv("LLM_BASE_URL"), None, "LLM_API_KEY"))
        self.model = self.model or default_model
        key = os.getenv(key_env) if key_env else "ollama"
        key = key or os.getenv("LLM_API_KEY") or "none"
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=os.getenv("LLM_BASE_URL", base), api_key=key, timeout=120)
        except Exception as e:  # pragma: no cover
            self.error = str(e)

    @property
    def params_b(self) -> Optional[float]:
        return MODEL_PARAMS.get(self.model or "")

    @property
    def available(self) -> bool:
        return self.client is not None and self.error is None

    def check(self) -> bool:
        if not self.client:
            return False
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=1)
            return True
        except Exception as e:
            self.error = str(e)[:300]
            return False

    def chat(self, messages: List[dict], temperature: float = 0.1, max_tokens: int = 700,
             json_mode: bool = False) -> str:
        if not self.client:
            raise RuntimeError("LLM chưa được cấu hình")
        kw = {}
        if json_mode and self.provider in ("ollama", "groq", "openrouter"):
            kw["response_format"] = {"type": "json_object"}
        t0 = time.time()
        r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature,
                                                max_tokens=max_tokens, **kw)
        self.last_latency = time.time() - t0
        return r.choices[0].message.content or ""

    def chat_json(self, messages: List[dict], **kw) -> Optional[dict]:
        txt = self.chat(messages, json_mode=True, **kw)
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
