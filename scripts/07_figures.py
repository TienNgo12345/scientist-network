"""Bước 7 – Vẽ hình cho báo cáo từ dữ liệu thật.

  reports/edges_by_type.png      số cạnh theo loại quan hệ (trục log, tô màu theo nguồn)
  reports/lineage_sommerfeld.png phả hệ học thuật 2 thế hệ của Arnold Sommerfeld
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from scinet.config import OUT, REPORTS

e = pd.read_csv(OUT / "edges.csv")

# ---- 1. Số cạnh theo loại quan hệ
NLP = {"BUILT_UPON", "CRITICIZED", "REFUTED", "PRIORITY_DISPUTE", "COLLABORATED_WITH", "ADVISOR_TO"}
vc = e.type.value_counts().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
cols = ["#DD8452" if t in NLP else "#4C72B0" for t in vc.index]
ax.barh(vc.index, vc.values, color=cols)
ax.set_xscale("log")
for i, v in enumerate(vc.values):
    ax.text(v * 1.08, i, f"{v:,}".replace(",", "."), va="center", fontsize=9)
ax.set_xlabel("Số cạnh (thang log)")
ax.set_xlim(1, vc.max() * 4)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4C72B0", label="Từ Infobox"), Patch(color="#DD8452", label="Từ văn bản (NLP)")],
          loc="lower right", frameon=False)
ax.set_title(f"{len(e):,} cạnh, {e.type.nunique()} loại quan hệ".replace(",", "."))
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(REPORTS / "edges_by_type.png", dpi=170); plt.close(fig)

# ---- 2. Phả hệ học thuật
ROOT = "Arnold Sommerfeld"
adv = e[e.type == "DOCTORAL_ADVISOR_OF"]
G = nx.DiGraph(); G.add_edges_from(zip(adv.source, adv.target))
gen1 = sorted(G.successors(ROOT)) if ROOT in G else []
T = nx.DiGraph()
for s in gen1:
    T.add_edge(ROOT, s)
    for g in sorted(G.successors(s)):
        T.add_edge(s, g)
level = {ROOT: 0}
for s in gen1: level[s] = 1
for n in T:
    level.setdefault(n, 2)
# bố cục: thế hệ 1 trên một vòng, thế hệ 2 quanh từng người thầy
import math
pos = {ROOT: (0, 0)}
k = len(gen1)
for i, s in enumerate(gen1):
    a = 2 * math.pi * i / k
    pos[s] = (math.cos(a), math.sin(a))
    kids = list(T.successors(s))
    for j, g in enumerate(kids):
        b = a + (j - (len(kids) - 1) / 2) * 0.09
        pos[g] = (1.75 * math.cos(b), 1.75 * math.sin(b))
fig, ax = plt.subplots(figsize=(13, 13))
nx.draw_networkx_edges(T, pos, ax=ax, edge_color="#B0B0B0", arrows=False, width=0.8)
nx.draw_networkx_nodes(T, pos, nodelist=[ROOT], node_color="#C44E52", node_size=900, ax=ax)
nx.draw_networkx_nodes(T, pos, nodelist=gen1, node_color="#4C72B0", node_size=260, ax=ax)
nx.draw_networkx_nodes(T, pos, nodelist=[n for n in T if level[n] == 2], node_color="#8FB3D9", node_size=60, ax=ax)
ax.text(0, -0.13, ROOT, ha="center", va="top", fontsize=13, fontweight="bold")
for s in gen1:
    x, y = pos[s]
    ax.text(x * 1.08, y * 1.08, s, fontsize=8.5, ha="left" if x > 0.1 else ("right" if x < -0.1 else "center"), va="center")
for n in T:
    if level[n] == 2:
        x, y = pos[n]
        ang = math.degrees(math.atan2(y, x))
        rot = ang if -90 <= ang <= 90 else ang + 180
        ax.text(x * 1.03, y * 1.03, n, fontsize=6, rotation=rot, rotation_mode="anchor",
                ha="left" if -90 <= ang <= 90 else "right", va="center", color="#333")
n2 = sum(1 for n in T if level[n] == 2)
ax.set_title(f"Phả hệ học thuật của {ROOT}: {len(gen1)} học trò tiến sĩ, {n2} học trò của học trò", fontsize=14)
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.axis("off")
fig.tight_layout(); fig.savefig(REPORTS / "lineage_sommerfeld.png", dpi=150, facecolor="white"); plt.close(fig)
print(len(gen1), n2)
