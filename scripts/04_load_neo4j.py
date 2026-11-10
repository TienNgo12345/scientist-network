"""Bước 4 – Nạp đồ thị vào Neo4j (kèm ràng buộc unique và index).

Yêu cầu: Neo4j 5.x đang chạy (xem docker-compose.yml), biến môi trường
NEO4J_URI (mặc định bolt://localhost:7687), NEO4J_USER, NEO4J_PASSWORD.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from neo4j import GraphDatabase

from scinet.config import OUT, RELATION_TYPES

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "scinet2026"))
LABELS = ["Scientist", "Institution", "Field", "Award"]


def chunks(df, n=1000):
    for i in range(0, len(df), n):
        yield df.iloc[i:i + n].to_dict("records")


def main():
    nodes = pd.read_csv(OUT / "nodes.csv").fillna("")
    edges = pd.read_csv(OUT / "edges.csv").fillna("")
    drv = GraphDatabase.driver(URI, auth=AUTH)
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        for lb in LABELS:
            s.run(f"CREATE CONSTRAINT {lb.lower()}_name IF NOT EXISTS "
                  f"FOR (n:{lb}) REQUIRE n.name IS UNIQUE")
        for lb in LABELS:
            part = nodes[nodes.label == lb]
            for rows in chunks(part):
                s.run(f"UNWIND $rows AS r MERGE (n:{lb} {{name: r.name}}) "
                      f"SET n.id = r.id, n.url = r.url, n.crawled = r.crawled", rows=rows)
        label_of = dict(zip(nodes.name, nodes.label))
        edges["sl"] = edges.source.map(label_of)
        edges["tl"] = edges.target.map(label_of)
        for (rt, sl, tl), part in edges.groupby(["type", "sl", "tl"]):
            assert rt in RELATION_TYPES, rt
            for rows in chunks(part):
                s.run(f"UNWIND $rows AS r "
                      f"MATCH (a:{sl} {{name: r.source}}), (b:{tl} {{name: r.target}}) "
                      f"MERGE (a)-[e:{rt}]->(b) "
                      f"SET e.method = r.method, e.confidence = r.confidence, "
                      f"e.evidence = r.evidence, e.section = r.section, e.page = r.page",
                      rows=rows)
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        m = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"Neo4j: {n} nút, {m} cạnh")
    drv.close()


if __name__ == "__main__":
    main()
