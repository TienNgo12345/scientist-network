"""Bước 5 – Thống kê mạng (NetworkX) + biểu đồ cho báo cáo.

Đầu ra: reports/stats.json, reports/stats.md, reports/*.png
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from scinet.config import OUT, REPORTS

SS_TYPES = ["DOCTORAL_ADVISOR_OF", "ACADEMIC_ADVISOR_OF", "INFLUENCED", "BUILT_UPON",
            "CRITICIZED", "REFUTED", "PRIORITY_DISPUTE", "COLLABORATED_WITH"]


def top(d, k=10):
    return [(n, round(v, 5)) for n, v in sorted(d.items(), key=lambda x: -x[1])[:k]]


def main():
    nodes = pd.read_csv(OUT / "nodes.csv")
    edges = pd.read_csv(OUT / "edges.csv")
    G = nx.MultiDiGraph()
    for r in nodes.itertuples():
        G.add_node(r.name, label=r.label)
    for r in edges.itertuples():
        G.add_edge(r.source, r.target, type=r.type)

    st = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
          "nodes_by_label": nodes.label.value_counts().to_dict(),
          "edges_by_type": edges.type.value_counts().to_dict(),
          "edges_by_method": edges.method.value_counts().to_dict(),
          "relation_types": int(edges.type.nunique())}

    # Đồ thị đơn vô hướng toàn mạng
    U = nx.Graph(G)
    degs = dict(U.degree())
    st["avg_degree"] = round(sum(degs.values()) / max(len(degs), 1), 3)
    st["density"] = round(nx.density(U), 6)
    comps = list(nx.connected_components(U))
    st["components"] = len(comps)
    lcc = max(comps, key=len) if comps else set()
    st["lcc_size"] = len(lcc)
    st["lcc_ratio"] = round(len(lcc) / max(len(U), 1), 4)

    # Mạng con nhà khoa học – nhà khoa học (trọng tâm đề tài)
    sci = set(nodes[nodes.label == "Scientist"].name)
    ssE = edges[edges.type.isin(SS_TYPES) & edges.source.isin(sci) & edges.target.isin(sci)]
    S = nx.DiGraph()
    S.add_edges_from(zip(ssE.source, ssE.target))
    Su = S.to_undirected()
    st["ss_nodes"], st["ss_edges"] = S.number_of_nodes(), S.number_of_edges()
    st["ss_avg_clustering"] = round(nx.average_clustering(Su), 4) if len(Su) else 0
    st["ss_reciprocity"] = round(nx.reciprocity(S), 4) if len(S) else 0
    try:
        st["ss_degree_assortativity"] = round(nx.degree_assortativity_coefficient(Su), 4)
    except Exception:
        st["ss_degree_assortativity"] = None
    if len(Su):
        lc = Su.subgraph(max(nx.connected_components(Su), key=len)).copy()
        st["ss_lcc_size"] = lc.number_of_nodes()
        if lc.number_of_nodes() <= 6000:
            st["ss_lcc_avg_shortest_path"] = round(nx.average_shortest_path_length(lc), 3)
            st["ss_lcc_diameter_approx"] = nx.approximation.diameter(lc)
        pr = nx.pagerank(S)
        bc = nx.betweenness_centrality(Su, k=min(500, len(Su)), seed=42)
        st["top_pagerank"] = top(pr)
        st["top_betweenness"] = top(bc)
        st["top_in_degree"] = top(dict(S.in_degree()))
        st["top_out_degree"] = top(dict(S.out_degree()))
        comms = nx.community.louvain_communities(Su, seed=42)
        st["louvain_communities"] = len(comms)
        st["louvain_modularity"] = round(nx.community.modularity(Su, comms), 4)
        st["largest_communities"] = [len(c) for c in sorted(comms, key=len, reverse=True)[:10]]

    # Mạng thầy – trò (cây phả hệ học thuật)
    adv = edges[edges.type == "DOCTORAL_ADVISOR_OF"]
    A = nx.DiGraph(); A.add_edges_from(zip(adv.source, adv.target))
    if len(A):
        st["advisor_dag"] = nx.is_directed_acyclic_graph(A)
        if st["advisor_dag"]:
            st["longest_academic_lineage"] = len(nx.dag_longest_path(A))
        st["top_doctoral_advisors"] = top(dict(A.out_degree()))

    (REPORTS / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Biểu đồ ------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    vc = edges.type.value_counts()
    vc.sort_values().plot.barh(color="#4C72B0")
    plt.xlabel("Số cạnh"); plt.title("Số cạnh theo loại quan hệ"); plt.tight_layout()
    plt.savefig(REPORTS / "edges_by_type.png", dpi=150); plt.close()

    cnt = Counter(degs.values())
    xs, ys = zip(*sorted(cnt.items()))
    plt.figure(figsize=(5, 4))
    plt.loglog(xs, ys, "o", ms=3)
    plt.xlabel("Bậc k"); plt.ylabel("Số nút có bậc k"); plt.title("Phân phối bậc (log-log)")
    plt.tight_layout(); plt.savefig(REPORTS / "degree_distribution.png", dpi=150); plt.close()

    # ---- Bảng markdown -------------------------------------------------------
    md = ["| Chỉ số | Giá trị |", "| --- | --- |"]
    for k, v in st.items():
        if not isinstance(v, (list, dict)):
            md.append(f"| {k} | {v} |")
    (REPORTS / "stats.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
