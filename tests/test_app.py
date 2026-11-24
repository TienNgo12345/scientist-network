"""Kiểm thử nhanh ứng dụng GraphRAG trên đồ thị cuối (không cần LLM).
Chạy: python tests/test_app.py   -> in OK nếu đạt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.graphrag import GraphRAG
from app.kg import KnowledgeGraph

kg = KnowledgeGraph()
rag = GraphRAG(kg, None)


def names(q):
    return [i["node"] for i in rag.ask(q, use_llm=False).items]


assert "Felix Klein" in names("Ai là thầy hướng dẫn của thầy hướng dẫn của Werner Heisenberg?")
assert set(names("Học trò của Arnold Sommerfeld đã đoạt giải Nobel Vật lý là ai?")) >= {"Werner Heisenberg", "Wolfgang Pauli", "Hans Bethe"}
assert "Ernst Mach" in names("Ai đã phê bình Isaac Newton?")
a = rag.ask("Albert Einstein và Richard Feynman liên hệ với nhau như thế nào?", use_llm=False)
assert a.plan.intent == "path" and a.items and len(a.items[0]["chains"][0]) <= 4
assert names("Những người đoạt huy chương Fields từng học ở đâu nhiều nhất?")[0] in ("Princeton University", "École normale supérieure (Paris)")
assert rag.ask("Các cộng đồng nghiên cứu lớn nhất trong mạng là gì?", use_llm=False).plan.intent == "global"
print("OK")
