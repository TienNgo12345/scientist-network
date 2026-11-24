"""Bước 13 – Các thuật toán phân tích mạng xã hội trên đồ thị cuối cùng.

1. Thế giới nhỏ: khoảng cách ngắn nhất trung bình (tính CHÍNH XÁC bằng BFS từ mọi đỉnh của thành
   phần liên thông lớn nhất), đường kính, phân bố số bước; so với mạng ngẫu nhiên Erdős–Rényi và
   mạng ngẫu nhiên giữ nguyên bậc (configuration model) cùng số đỉnh/cạnh -> hệ số sigma.
2. Xếp hạng: PageRank (có hướng), betweenness (xấp xỉ, k=2000 nguồn), bậc, HITS (hub/authority).
3. Cộng đồng + hạt nhân: Louvain (modularity), gán nhãn cộng đồng theo lĩnh vực/viện phổ biến;
   phân rã k-core -> lõi lớn nhất ("hạt nhân" của mạng) và lõi của từng cộng đồng.

Chạy trên 2 mạng: (a) đồ thị tri thức đầy đủ (vô hướng), (b) mạng nhà khoa học – nhà khoa học.
Đầu ra: reports/sna.json, reports/sna.md, reports/fig_*.png, data/final/node_metrics.csv,
        data/final/communities.json
Chạy  : python scripts/13_sna_algorithms.py [--quick]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from scinet.config import DATA, ROOT

FINAL, REPORTS = DATA / "final", ROOT / "reports"
SCI_REL = {"DOCTORAL_ADVISOR_OF", "ACADEMIC_ADVISOR_OF", "INFLUENCED", "COLLABORATED_WITH", "CRITICIZED",
           "REFUTED", "BUILT_UPON", "PRIORITY_DISPUTE"}


def load():
    nodes = {r["name"]: r for r in csv.DictReader(open(FINAL / "nodes.csv", encoding="utf-8"))}
    edges = list(csv.DictReader(open(FINAL / "edges.csv", encoding="utf-8")))
    D = nx.MultiDiGraph()
    for n, r in nodes.items():
        D.add_node(n, label=r["label"])
    for e in edges:
        D.add_edge(e["source"], e["target"], type=e["type"])
    return nodes, edges, D


def apl_exact(G):
    """Khoảng cách trung bình + đường kính + phân bố số bước, chính xác (BFS mọi đỉnh, theo khối)."""
    idx = {n: i for i, n in enumerate(G.nodes())}
    rows, cols = [], []
    for u, v in G.edges():
        rows += [idx[u], idx[v]]; cols += [idx[v], idx[u]]
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(idx), len(idx)))
    n = len(idx); total = 0.0; cnt = 0; diam = 0; hist = Counter()
    for s in range(0, n, 400):
        d = shortest_path(A, unweighted=True, directed=False, indices=list(range(s, min(n, s + 400))))
        d = d[np.isfinite(d) & (d > 0)].astype(int)
        total += d.sum(); cnt += d.size; diam = max(diam, int(d.max()))
        for h, c in enumerate(np.bincount(d)):
            if c:
                hist[h] += int(c)
    hist = {k: int(v) for k, v in sorted(hist.items()) if v}
    return total / cnt, diam, hist


def small_world(G, name, n_rand=3, seed=0):
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    n, m = lcc.number_of_nodes(), lcc.number_of_edges()
    L, diam, hist = apl_exact(lcc)
    C = nx.average_clustering(lcc)
    Ct = nx.transitivity(lcc)
    k = 2 * m / n
    er_L, er_C, cm_L, cm_C = [], [], [], []
    for i in range(n_rand):
        er = nx.gnm_random_graph(n, m, seed=seed + i)
        er = er.subgraph(max(nx.connected_components(er), key=len))
        er_L.append(apl_exact(er)[0]); er_C.append(nx.average_clustering(er))
        degs = [d for _, d in lcc.degree()]
        cm = nx.Graph(nx.configuration_model(degs, seed=seed + i)); cm.remove_edges_from(nx.selfloop_edges(cm))
        cm = cm.subgraph(max(nx.connected_components(cm), key=len))
        cm_L.append(apl_exact(cm)[0]); cm_C.append(nx.average_clustering(cm))
    Lr, Cr = float(np.mean(er_L)), float(np.mean(er_C))
    within6 = sum(v for h, v in hist.items() if h <= 6) / sum(hist.values())
    return {
        "network": name, "lcc_nodes": n, "lcc_edges": m, "lcc_share": round(n / G.number_of_nodes(), 4),
        "avg_degree": round(k, 3), "avg_shortest_path": round(L, 3), "diameter": diam,
        "avg_clustering": round(C, 4), "transitivity": round(Ct, 4),
        "log_n_over_log_k": round(np.log(n) / np.log(k), 3) if k > 1 else None,
        "random_ER": {"avg_shortest_path": round(Lr, 3), "avg_clustering": round(Cr, 5)},
        "random_config": {"avg_shortest_path": round(float(np.mean(cm_L)), 3),
                          "avg_clustering": round(float(np.mean(cm_C)), 5)},
        "sigma": round((C / Cr) / (L / Lr), 2) if Cr > 0 else None,
        "share_pairs_within_6_hops": round(within6, 4), "hop_histogram": hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    nodes, edges, D = load()
    U = nx.Graph(D.to_undirected())
    sci = nx.Graph([(e["source"], e["target"]) for e in edges if e["type"] in SCI_REL])
    res = {"graph": {"nodes": D.number_of_nodes(), "edges": D.number_of_edges(), "undirected_simple_edges": U.number_of_edges()}}

    # ---------------- 1. thế giới nhỏ ----------------
    res["small_world"] = [small_world(U, "Đồ thị tri thức đầy đủ", 1 if args.quick else 3),
                          small_world(sci, "Mạng nhà khoa học – nhà khoa học", 1 if args.quick else 3)]
    for s in res["small_world"]:
        print({k: v for k, v in s.items() if k != "hop_histogram"})

    # ---------------- 2. xếp hạng ----------------
    Dsimple = nx.DiGraph(D)
    pr = nx.pagerank(Dsimple, alpha=0.85)
    bt = nx.betweenness_centrality(U, k=300 if args.quick else 2000, seed=42)
    hubs, auth = nx.hits(nx.DiGraph([(e["source"], e["target"]) for e in edges if e["type"] in SCI_REL]),
                         max_iter=500)
    deg = dict(U.degree())
    pr_sci = nx.pagerank(nx.DiGraph([(e["source"], e["target"]) for e in edges if e["type"] in SCI_REL]))

    def top(d, k=20, label=None):
        it = [(n, v) for n, v in d.items() if label is None or nodes[n]["label"] == label]
        return [(n, round(v, 6)) for n, v in sorted(it, key=lambda x: -x[1])[:k]]
    res["ranking"] = {
        "pagerank_all_nodes": top(pr), "pagerank_scientists": top(pr, label="Scientist"),
        "pagerank_institutions": top(pr, label="Institution"), "pagerank_awards": top(pr, label="Award"),
        "pagerank_scientist_network": top(pr_sci),
        "betweenness_scientists": top(bt, label="Scientist"), "degree_scientists": top(deg, label="Scientist"),
        "hits_authority": top(auth, 15), "hits_hub": top(hubs, 15),
    }
    sp = [n for n in nodes if nodes[n]["label"] == "Scientist"]
    from scipy.stats import spearmanr
    res["ranking"]["spearman_pagerank_vs_betweenness_scientists"] = round(
        float(spearmanr([pr[n] for n in sp], [bt.get(n, 0) for n in sp]).correlation), 3)

    # ---------------- 3. cộng đồng + hạt nhân ----------------
    comms = nx.community.louvain_communities(U, seed=42, resolution=1.0)
    comms = sorted(comms, key=len, reverse=True)
    mod = nx.community.modularity(U, comms)
    lpa = list(nx.community.label_propagation_communities(U))
    cid = {n: i for i, c in enumerate(comms) for n in c}
    core = nx.core_number(U)
    kmax = max(core.values())
    main_core = [n for n, c in core.items() if c == kmax]
    comm_info = []
    for i, c in enumerate(comms[:40]):
        lab = Counter(nodes[n]["label"] for n in c)
        fields = Counter(n for n in c if nodes[n]["label"] == "Field")
        # chủ đề cộng đồng: lĩnh vực / viện / giải có nhiều cạnh nhất bên trong cộng đồng
        def best(label):
            cand = [(n, deg[n]) for n in c if nodes[n]["label"] == label]
            return [n for n, _ in sorted(cand, key=lambda x: -x[1])[:3]]
        sub = U.subgraph(c)
        ccore = nx.core_number(sub)
        ck = max(ccore.values())
        nucleus = sorted([n for n, v in ccore.items() if v == ck and nodes[n]["label"] == "Scientist"],
                         key=lambda n: -pr[n])[:8]
        comm_info.append({"id": i, "size": len(c), "labels": dict(lab), "top_fields": best("Field"),
                          "top_institutions": best("Institution"), "top_awards": best("Award"),
                          "top_scientists_pagerank": [n for n in sorted(c, key=lambda n: -pr[n]) if nodes[n]["label"] == "Scientist"][:8],
                          "k_core_max": ck, "nucleus": nucleus})
    shells = Counter(core.values())
    res["communities"] = {"louvain_count": len(comms), "louvain_modularity": round(mod, 4),
                          "label_propagation_count": len(lpa),
                          "label_propagation_modularity": round(nx.community.modularity(U, lpa), 4),
                          "sizes_top20": [len(c) for c in comms[:20]],
                          "share_in_top10": round(sum(len(c) for c in comms[:10]) / U.number_of_nodes(), 4),
                          "top": comm_info}
    res["k_core"] = {"k_max": kmax, "main_core_size": len(main_core),
                     "main_core_by_label": dict(Counter(nodes[n]["label"] for n in main_core)),
                     "main_core_scientists": sorted([n for n in main_core if nodes[n]["label"] == "Scientist"],
                                                    key=lambda n: -pr[n])[:40],
                     "main_core_other": sorted([n for n in main_core if nodes[n]["label"] != "Scientist"],
                                               key=lambda n: -pr[n])[:30],
                     "shell_sizes": dict(sorted(shells.items()))}
    rc = nx.rich_club_coefficient(U, normalized=False)
    res["k_core"]["rich_club_k20"] = round(rc.get(20, 0), 4)

    # ---------------- lưu số đo từng nút cho ứng dụng ----------------
    sci_core = nx.core_number(sci)
    with open(FINAL / "node_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "label", "degree", "in_degree", "out_degree", "pagerank", "pagerank_rank",
                    "betweenness", "community", "core_number", "sci_core_number", "hits_authority", "hits_hub"])
        rank = {n: i + 1 for i, (n, _) in enumerate(sorted(pr.items(), key=lambda x: -x[1]))}
        for n in nodes:
            w.writerow([n, nodes[n]["label"], deg.get(n, 0), Dsimple.in_degree(n), Dsimple.out_degree(n),
                        f"{pr[n]:.8f}", rank[n], f"{bt.get(n, 0):.8f}", cid.get(n, -1), core.get(n, 0),
                        sci_core.get(n, 0), f"{auth.get(n, 0):.8f}", f"{hubs.get(n, 0):.8f}"])
    json.dump(comm_info, open(FINAL / "communities.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(res, open(REPORTS / "sna.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # ---------------- hình ----------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, s in zip(axs, res["small_world"]):
        h = s["hop_histogram"]; tot = sum(h.values())
        ax.bar(list(h.keys()), [v / tot for v in h.values()], color="#3B6EA8")
        ax.axvline(s["avg_shortest_path"], color="#C0392B", ls="--", label=f"L = {s['avg_shortest_path']}")
        ax.axvline(s["random_ER"]["avg_shortest_path"], color="#7F8C8D", ls=":",
                   label=f"L ngẫu nhiên = {s['random_ER']['avg_shortest_path']}")
        ax.set_title(s["network"]); ax.set_xlabel("Số bước giữa hai đỉnh"); ax.set_ylabel("Tỉ lệ cặp đỉnh")
        ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(REPORTS / "fig_small_world.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ks = sorted(shells); ax.bar(ks, [shells[k] for k in ks], color="#5D8C3E")
    ax.set_yscale("log"); ax.set_xlabel("k (core number)"); ax.set_ylabel("Số đỉnh (log)")
    ax.set_title(f"Phân rã k-core – lõi lớn nhất k = {kmax} gồm {len(main_core)} đỉnh")
    fig.tight_layout(); fig.savefig(REPORTS / "fig_kcore.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = res["ranking"]["pagerank_scientists"][:15][::-1]
    ax.barh([n for n, _ in t], [v * 1000 for _, v in t], color="#8E44AD")
    ax.set_xlabel("PageRank (×10⁻³)"); ax.set_title("15 nhà khoa học có PageRank cao nhất")
    fig.tight_layout(); fig.savefig(REPORTS / "fig_pagerank.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    sizes = [len(c) for c in comms]
    ax.bar(range(1, 31), sizes[:30], color="#D68910")
    ax.set_xlabel("Cộng đồng (xếp theo kích thước)"); ax.set_ylabel("Số đỉnh")
    ax.set_title(f"Louvain: {len(comms)} cộng đồng, modularity Q = {mod:.3f}")
    fig.tight_layout(); fig.savefig(REPORTS / "fig_communities.png", dpi=200); plt.close(fig)

    print(json.dumps({"ranking_top5": res["ranking"]["pagerank_scientists"][:5],
                      "communities": {k: v for k, v in res["communities"].items() if k != "top"},
                      "k_core": {k: v for k, v in res["k_core"].items() if k not in ("main_core_scientists", "main_core_other")}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
