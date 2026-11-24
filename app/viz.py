"""Vẽ đồ thị tương tác (vis-network qua pyvis, nhúng JS trực tiếp -> chạy được khi offline)."""
from typing import Iterable, List, Optional

from pyvis.network import Network

from .kg import LABEL_COLOR, Hop, KnowledgeGraph

REL_SHORT = {"DOCTORAL_ADVISOR_OF": "hướng dẫn TS", "ACADEMIC_ADVISOR_OF": "cố vấn", "INFLUENCED": "ảnh hưởng",
             "COLLABORATED_WITH": "hợp tác", "CRITICIZED": "phê bình", "REFUTED": "bác bỏ", "BUILT_UPON": "kế thừa",
             "PRIORITY_DISPUTE": "tranh chấp", "WORKED_AT": "làm việc", "STUDIED_AT": "học tại",
             "ADVISOR_TO": "cố vấn cho", "RECEIVED_AWARD": "giải", "WORKS_IN_FIELD": "lĩnh vực"}
REL_COLOR = {"DOCTORAL_ADVISOR_OF": "#1f4e79", "ACADEMIC_ADVISOR_OF": "#5b8db8", "INFLUENCED": "#8e44ad",
             "COLLABORATED_WITH": "#16a085", "CRITICIZED": "#c0392b", "REFUTED": "#922b21", "BUILT_UPON": "#d35400",
             "PRIORITY_DISPUTE": "#7d3c98"}


def _safe(s: str) -> str:
    return (s or "").replace("<", "‹").replace(">", "›")


def render(kg: KnowledgeGraph, nodes: Iterable[str], highlight: Optional[List[str]] = None,
           path_edges: Optional[List[Hop]] = None, height: int = 560, hierarchical: bool = False,
           only_edges: Optional[List[Hop]] = None) -> str:
    nodes = list(dict.fromkeys(nodes))
    highlight = set(highlight or [])
    net = Network(height=f"{height}px", width="100%", directed=True, cdn_resources="in_line",
                  bgcolor="#ffffff", font_color="#1d1d1f")
    for n in nodes:
        lab = kg.nodes.get(n, {}).get("label", "Scientist")
        pr = float(kg.metrics.get(n, {}).get("pagerank", 0) or 0)
        size = 12 + min(28, pr * 30000)
        if n in highlight:
            size += 10
        title = f"{n}\n{lab}\nPageRank hạng {kg.metrics.get(n, {}).get('pagerank_rank', '?')}"
        net.add_node(n, label=_safe(n if len(n) < 28 else n[:26] + "…"), title=_safe(title), size=size,
                     color={"background": LABEL_COLOR.get(lab, "#999"), "border": "#111" if n in highlight else "#fff"},
                     borderWidth=3 if n in highlight else 1, shape="dot",
                     font={"size": 15 if n in highlight else 12, "face": "Inter, Arial"})
    keep = set(nodes)
    pe = {(h.source, h.target, h.rel) for h in (path_edges or [])}
    edges = only_edges if only_edges is not None else [
        Hop(u, d["type"], v, d.get("evidence", "")) for u, v, d in kg.G.subgraph(nodes).edges(data=True)]
    seen = set()
    for h in edges:
        if h.source not in keep or h.target not in keep:
            continue
        k = (h.source, h.target, h.rel)
        if k in seen:
            continue
        seen.add(k)
        on = k in pe
        net.add_edge(h.source, h.target, label=REL_SHORT.get(h.rel, h.rel) if on or len(edges) < 60 else "",
                     title=_safe(h.rel + (": " + h.evidence[:300] if h.evidence else "")),
                     color="#e67e22" if on else REL_COLOR.get(h.rel, "#b8bec6"), width=4 if on else 1.4,
                     arrows="to", font={"size": 10, "align": "middle", "color": "#555"})
    opts = {
        "physics": {"barnesHut": {"gravitationalConstant": -9000, "springLength": 140, "springConstant": 0.03},
                    "stabilization": {"iterations": 250}},
        "interaction": {"hover": True, "navigationButtons": True, "tooltipDelay": 120},
        "edges": {"smooth": {"type": "dynamic"}},
    }
    if hierarchical:
        opts["layout"] = {"hierarchical": {"enabled": True, "direction": "LR", "sortMethod": "directed",
                                           "levelSeparation": 260, "nodeSpacing": 34, "treeSpacing": 60}}
        opts["physics"] = {"enabled": False}
    import json
    net.set_options(json.dumps(opts))
    html = net.generate_html()
    # luôn căn đồ thị vào giữa khung sau khi vẽ xong
    return html.replace("return network;", "network.once('afterDrawing', function(){network.fit();});\n                  return network;", 1)
