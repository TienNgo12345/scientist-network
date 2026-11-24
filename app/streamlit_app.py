"""SciNet Explorer – ứng dụng đồ thị tri thức các nhà khoa học.

Chạy:  streamlit run app/streamlit_app.py
Trang:
  1. Hỏi đáp GraphRAG   – chatbot suy luận nhiều bước trên đồ thị, hiện chuỗi suy luận + đồ thị bằng chứng
  2. Chân dung nhà khoa học – toàn cảnh một nút: số đo mạng, mạng lân cận, phả hệ, cộng đồng, bằng chứng
  3. Tổng quan mạng      – thế giới nhỏ, xếp hạng, cộng đồng, hạt nhân, chất lượng đồ thị
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.graphrag import GraphRAG
from app.kg import LABEL_COLOR, REL_VI, KnowledgeGraph
from app.llm import LLM, MODEL_PARAMS
from app.viz import render

st.set_page_config(page_title="SciNet Explorer", page_icon="🔭", layout="wide")


def show_html(html: str, height: int):
    """Nhúng đồ thị vis-network (HTML do chính ứng dụng sinh ra, đã lọc ký tự < >)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # Streamlit cũ
        import streamlit.components.v1 as components
        components.html(html, height=height)

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {background: linear-gradient(120deg,#0f2a4a 0%,#1f4e79 55%,#2e7d6b 100%); color:#fff; padding:22px 28px;
       border-radius:16px; margin-bottom:18px;}
.hero h1 {color:#fff; margin:0; font-size:1.9rem;} .hero p {margin:6px 0 0; opacity:.88;}
.kpi {background:#fff; border:1px solid #e6e8eb; border-radius:14px; padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,.04);}
.kpi .v {font-size:1.55rem; font-weight:700; color:#0f2a4a;} .kpi .l {font-size:.8rem; color:#6b7280; text-transform:uppercase; letter-spacing:.04em;}
.kpi .s {font-size:.78rem; color:#2e7d6b;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; margin:2px 4px 2px 0; color:#fff;}
.chain {background:#f6f8fa; border-left:4px solid #e67e22; padding:8px 12px; border-radius:6px; margin:4px 0; font-size:.9rem;}
.evid {background:#fbfbf7; border:1px solid #eee7cf; padding:8px 12px; border-radius:8px; margin:4px 0; font-size:.86rem; color:#444;}
.small {font-size:.82rem; color:#6b7280;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang nạp đồ thị tri thức…")
def load_kg():
    return KnowledgeGraph()


@st.cache_data
def load_json(name):
    p = ROOT / "reports" / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


kg = load_kg()
SNA, QUAL, QAE = load_json("sna.json"), load_json("quality.json"), load_json("qa_eval.json")


def kpi(col, value, label, sub=""):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def badge(text, color):
    return f"<span class='badge' style='background:{color}'>{text}</span>"


# ------------------------------------------------------------------ thanh bên: cấu hình LLM
with st.sidebar:
    st.markdown("### 🔭 SciNet Explorer")
    page = st.radio("Trang", ["💬 Hỏi đáp GraphRAG", "👤 Chân dung nhà khoa học", "📊 Tổng quan mạng"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**Mô hình ngôn ngữ (≤ 9B)**")
    import os
    from app.llm import _load_env
    _load_env()
    provs = ["ollama", "openrouter", "groq", "none"]
    env_p = os.getenv("LLM_PROVIDER", "ollama").lower()
    provider = st.selectbox("Nhà cung cấp", provs, index=provs.index(env_p) if env_p in provs else 0,
                            help="ollama: chạy cục bộ (ollama pull qwen2.5:7b). none: không dùng LLM.")
    model = st.text_input("Model", value=os.getenv("LLM_MODEL") if os.getenv("LLM_PROVIDER", "").lower() == provider and os.getenv("LLM_MODEL") else {"ollama": "qwen2.5:7b", "openrouter": "meta-llama/llama-3.1-8b-instruct",
                                          "groq": "llama-3.1-8b-instant", "none": "-"}[provider])
    if "llm_key" not in st.session_state or st.session_state.llm_key != (provider, model):
        st.session_state.llm = LLM(provider, model if provider != "none" else None)
        st.session_state.llm_key = (provider, model)
        st.session_state.llm_ok = None
    llm: LLM = st.session_state.llm
    pb = MODEL_PARAMS.get(model)
    if pb:
        st.caption(f"Kích thước: **{pb} tỷ tham số** " + ("✅ ≤ 9B" if pb <= 9.5 else "⚠️ vượt 9B"))
    if provider != "none" and st.button("Kiểm tra kết nối"):
        st.session_state.llm_ok = llm.check()
    if st.session_state.get("llm_ok") is True:
        st.success("LLM sẵn sàng")
    elif st.session_state.get("llm_ok") is False:
        st.error("Không kết nối được LLM – ứng dụng dùng bộ lập kế hoạch theo luật.")
        st.caption(str(llm.error)[:200])
    use_llm = st.toggle("Dùng LLM để lập kế hoạch + viết câu trả lời", value=provider != "none")
    st.divider()
    g = QUAL.get("output", {})
    st.caption(f"Đồ thị: **{len(kg.nodes):,}** nút · **{len(kg.edges):,}** cạnh · "
               f"**{g.get('relation_types_with_data', 13)}** loại quan hệ")

rag = GraphRAG(kg, llm if (use_llm and st.session_state.get("llm_ok") is not False) else None)

# ================================================================== 1. HỎI ĐÁP
if page.startswith("💬"):
    st.markdown("<div class='hero'><h1>Hỏi đáp suy luận nhiều bước trên đồ thị</h1>"
                "<p>GraphRAG: liên kết thực thể → lập kế hoạch truy vấn → duyệt đồ thị nhiều bước → "
                "truy hồi bằng chứng Wikipedia → mô hình ngôn ngữ nhỏ viết câu trả lời</p></div>",
                unsafe_allow_html=True)
    examples = ["Ai là thầy hướng dẫn của thầy hướng dẫn của Werner Heisenberg?",
                "Học trò của Arnold Sommerfeld đã đoạt giải Nobel Vật lý là ai?",
                "Những người đoạt huy chương Fields từng học ở đâu nhiều nhất?",
                "Albert Einstein và Richard Feynman liên hệ với nhau như thế nào?",
                "Ai đã phê bình Isaac Newton?",
                "Các cộng đồng nghiên cứu lớn nhất trong mạng là gì?"]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.pending = ex
    if "history" not in st.session_state:
        st.session_state.history = []
    q = st.chat_input("Đặt câu hỏi về các nhà khoa học, thầy–trò, giải thưởng, viện nghiên cứu…")
    q = q or st.session_state.pop("pending", None)
    if q:
        with st.spinner("Đang suy luận trên đồ thị…"):
            st.session_state.history.append(rag.ask(q, use_llm=use_llm))
    for a in reversed(st.session_state.history[-6:]):
        with st.chat_message("user"):
            st.write(a.question)
        with st.chat_message("assistant"):
            st.markdown(a.text)
            src = "LLM " + (llm.model or "") if a.llm_used else "mẫu câu (không LLM)"
            st.caption(f"Kế hoạch: **{a.plan.source}** · ý định **{a.plan.intent}** · "
                       f"{len(a.items)} đáp án · câu trả lời: {src}")
            t1, t2, t3, t4 = st.tabs(["🧭 Chuỗi suy luận", "🕸️ Đồ thị bằng chứng", "📄 Bằng chứng Wikipedia", "🧾 Kế hoạch JSON"])
            with t1:
                st.markdown("**Thực thể nhận ra:** " + (", ".join(a.linked) or "—"))
                if a.trace:
                    st.dataframe(pd.DataFrame(a.trace), hide_index=True, width="stretch")
                for it in a.items[:8]:
                    for ch in it["chains"][:1]:
                        st.markdown("<div class='chain'>" + " → ".join(kg.describe(h) for h in ch) + "</div>",
                                    unsafe_allow_html=True)
            with t2:
                path_edges = [h for it in a.items[:12] for ch in it["chains"][:2] for h in ch]
                if a.subgraph_nodes:
                    show_html(render(kg, a.subgraph_nodes[:80], highlight=a.plan.entities,
                                           path_edges=path_edges, only_edges=path_edges or None, height=520),
                                    height=540)
                else:
                    st.info("Không có đồ thị con cho câu hỏi này.")
            with t3:
                for e in a.evidence[:8]:
                    st.markdown(f"<div class='evid'>“{e}”</div>", unsafe_allow_html=True)
                if not a.evidence:
                    st.caption("Các cạnh dùng trong suy luận lấy từ Infobox (không có câu văn bản).")
            with t4:
                st.json(a.plan.to_json())

# ================================================================== 2. CHÂN DUNG
elif page.startswith("👤"):
    st.markdown("<div class='hero'><h1>Chân dung toàn cảnh một nhà khoa học</h1>"
                "<p>Vị trí trong mạng, phả hệ học thuật, cộng đồng, hạt nhân và các quan hệ có bằng chứng</p></div>",
                unsafe_allow_html=True)
    sci = [n for n, r in kg.nodes.items() if r["label"] == "Scientist"]
    top = sorted(sci, key=lambda n: int(kg.metrics.get(n, {}).get("pagerank_rank", 10 ** 6)))
    default = top.index("Niels Bohr") if "Niels Bohr" in top else 0
    name = st.selectbox("Chọn nhà khoa học (gõ để tìm)", top, index=default)
    m = kg.metrics.get(name, {})
    node = kg.nodes[name]
    fields = [h.target for h in kg.neighbors(name, "WORKS_IN_FIELD", "out")]
    insts = [h.target for h in kg.neighbors(name, "WORKED_AT", "out")]
    awards = [h.target for h in kg.neighbors(name, "RECEIVED_AWARD", "out")]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"## {name}")
        st.markdown("".join(badge(f, LABEL_COLOR["Field"]) for f in fields[:5]) +
                    "".join(badge(i, LABEL_COLOR["Institution"]) for i in insts[:4]), unsafe_allow_html=True)
        if node.get("url"):
            st.markdown(f"[Trang Wikipedia]({node['url']})")
    with c2:
        st.markdown("".join(badge(a, LABEL_COLOR["Award"]) for a in awards[:8]) or "<span class='small'>Chưa có giải thưởng trong đồ thị</span>",
                    unsafe_allow_html=True)
    sci_bt = sorted(sci, key=lambda n: -float(kg.metrics.get(n, {}).get("betweenness", 0) or 0))
    bt_rank = sci_bt.index(name) + 1 if name in sci_bt else None
    sci_rank = top.index(name) + 1
    k = st.columns(6)
    kpi(k[0], f"#{sci_rank}", "PageRank", f"trên {len(sci):,} nhà khoa học")
    kpi(k[1], f"#{bt_rank}", "Betweenness", "vai trò cầu nối")
    kpi(k[2], m.get("degree", "-"), "Bậc", f"vào {m.get('in_degree')} · ra {m.get('out_degree')}")
    kpi(k[3], m.get("core_number", "-"), "k-core", f"lõi lớn nhất k = {SNA.get('k_core', {}).get('k_max', '?')}")
    kpi(k[4], m.get("community", "-"), "Cộng đồng", "Louvain")
    stud = len({c[-1] for c in kg.lineage(name, up=False, depth=1)})
    desc = len({c[-1] for c in kg.lineage(name, up=False, depth=3)})
    kpi(k[5], stud, "Học trò", f"{desc} hậu duệ ≤ 3 đời")
    st.write("")
    tabs = st.tabs(["🕸️ Mạng lân cận", "🌳 Phả hệ học thuật", "👥 Cộng đồng & hạt nhân", "📋 Quan hệ & bằng chứng",
                    "🧬 Nhà khoa học tương tự", "📝 Tóm tắt"])
    with tabs[0]:
        r = st.slider("Bán kính", 1, 2, 1)
        sub = kg.ego([name], radius=r, max_nodes=60 if r == 1 else 90)
        show_html(render(kg, list(sub.nodes()), highlight=[name], height=560), height=580)
        st.caption("Màu nút: " + " · ".join(f"<span style='color:{c}'>●</span> {l}" for l, c in LABEL_COLOR.items()),
                   unsafe_allow_html=True)
    with tabs[1]:
        up = kg.lineage(name, up=True, depth=5)
        down = kg.lineage(name, up=False, depth=2)
        prk = lambda n: float(kg.metrics.get(n, {}).get("pagerank", 0) or 0)
        kids = sorted({c[1] for c in down if len(c) == 2}, key=lambda n: -prk(n))
        show_kids = kids[:20]
        grand = []
        for kd in show_kids[:4]:
            gk = sorted({c[2] for c in down if len(c) == 3 and c[1] == kd}, key=lambda n: -prk(n))[:6]
            grand += gk
        anc = {x for c in up for x in c}
        nodes = {name} | anc | set(show_kids) | set(grand)
        edges = [h for n in nodes for r in ("DOCTORAL_ADVISOR_OF", "ACADEMIC_ADVISOR_OF")
                 for h in kg.neighbors(n, r, "out") if h.target in nodes]
        st.markdown(f"**{len(anc) - 1 if anc else 0} tổ tiên học thuật** (≤ 5 đời) · **{len(kids)} học trò** · "
                    f"**{len({c[-1] for c in down if len(c) == 3})} học trò của học trò**"
                    + (f" · hiển thị 20 học trò và học trò của 4 người nổi bật nhất" if len(kids) > 20 or grand else ""))
        if len(nodes) > 1:
            show_html(render(kg, list(nodes), highlight=[name], only_edges=edges, hierarchical=True,
                             height=620), height=640)
        longest = max(up, key=len) if up else [name]
        st.markdown("**Chuỗi thầy dài nhất:** " + " ← ".join(longest))
    with tabs[2]:
        c = kg.community_of(name)
        if c:
            st.markdown(f"**Cộng đồng {c['id']}** – {c['size']} nút · lĩnh vực: {', '.join(c['top_fields'])}")
            summ = kg.community_summaries.get(str(c["id"]))
            if summ:
                st.info(summ)
            a1, a2 = st.columns(2)
            a1.markdown("**Trung tâm (PageRank)**\n\n" + "\n".join(f"{i + 1}. {n}" for i, n in enumerate(c["top_scientists_pagerank"])))
            a2.markdown(f"**Hạt nhân (k-core = {c['k_core_max']})**\n\n" + "\n".join(f"- {n}" for n in c["nucleus"]))
            show_html(render(kg, list(dict.fromkeys(c["nucleus"] + c["top_scientists_pagerank"][:6] + [name])),
                                   highlight=[name], height=420), height=440)
    with tabs[3]:
        rows = []
        for h in kg.neighbors(name, None, "both"):
            fw, bw = REL_VI.get(h.rel, (h.rel, h.rel))
            rows.append({"Quan hệ": fw if h.source == name else bw,
                         "Đối tượng": h.target if h.source == name else h.source, "Loại": h.rel,
                         "Nguồn": h.method or "infobox", "Bằng chứng": h.evidence})
        df = pd.DataFrame(rows)
        if not df.empty:
            fig = px.bar(df["Loại"].value_counts().reset_index(), x="count", y="Loại", orientation="h",
                         color_discrete_sequence=["#1f4e79"], height=280)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="", xaxis_title="số cạnh")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df, hide_index=True, width="stretch", height=380)
    with tabs[4]:
        nb = set(kg.U.neighbors(name))
        sims = []
        for o in sci:
            if o == name:
                continue
            on = set(kg.U.neighbors(o))
            inter = nb & on
            if len(inter) >= 2:
                sims.append((o, len(inter) / len(nb | on), sorted(inter)[:5]))
        sims.sort(key=lambda x: -x[1])
        st.caption("Độ tương tự Jaccard trên tập láng giềng (cùng thầy, cùng viện, cùng giải, cùng lĩnh vực…)")
        st.dataframe(pd.DataFrame([{"Nhà khoa học": o, "Jaccard": round(s, 3), "Điểm chung": ", ".join(i)}
                                   for o, s, i in sims[:15]]), hide_index=True, width="stretch")
    with tabs[5]:
        a = GraphRAG(kg, None).ask(f"Kể về {name}", use_llm=False)
        if use_llm and llm.available and st.button("✍️ Viết tóm tắt bằng LLM"):
            with st.spinner("LLM đang viết…"):
                try:
                    txt = llm.chat([{"role": "system", "content": "Viết một đoạn chân dung 150-200 từ tiếng Việt về nhà "
                                     "khoa học, CHỈ dựa vào dữ kiện đồ thị được cho, nêu vị trí trong mạng."},
                                    {"role": "user", "content": a.context + "\n" + a.text}], max_tokens=500)
                    st.success(txt)
                except Exception as e:
                    st.error(str(e)[:200])
        st.markdown(a.text)

# ================================================================== 3. TỔNG QUAN
else:
    st.markdown("<div class='hero'><h1>Tổng quan mạng xã hội các nhà khoa học</h1>"
                "<p>Thế giới nhỏ · xếp hạng PageRank · cộng đồng Louvain · hạt nhân k-core · chất lượng đồ thị</p></div>",
                unsafe_allow_html=True)
    sw = SNA.get("small_world", [{}])[0]
    k = st.columns(6)
    kpi(k[0], f"{len(kg.nodes):,}", "Nút", "4 loại thực thể")
    kpi(k[1], f"{len(kg.edges):,}", "Cạnh", f"{QUAL.get('output', {}).get('relation_types_with_data', 13)} loại quan hệ")
    kpi(k[2], sw.get("avg_shortest_path", "-"), "Khoảng cách TB", f"ngẫu nhiên {sw.get('random_ER', {}).get('avg_shortest_path', '-')}")
    kpi(k[3], f"{100 * sw.get('share_pairs_within_6_hops', 0):.1f}%", "Cặp ≤ 6 bước", "“sáu bậc phân cách”")
    kpi(k[4], SNA.get("communities", {}).get("louvain_count", "-"), "Cộng đồng",
        f"Q = {SNA.get('communities', {}).get('louvain_modularity', '-')}")
    kpi(k[5], SNA.get("k_core", {}).get("main_core_size", "-"), "Hạt nhân", f"k-core k = {SNA.get('k_core', {}).get('k_max', '-')}")
    st.write("")
    t = st.tabs(["🌐 Thế giới nhỏ", "🏆 Xếp hạng", "👥 Cộng đồng", "🎯 Hạt nhân", "✅ Chất lượng & đánh giá"])
    with t[0]:
        cols = st.columns(2)
        for col, s in zip(cols, SNA.get("small_world", [])):
            h = {int(a): b for a, b in s["hop_histogram"].items()}
            tot = sum(h.values())
            fig = go.Figure(go.Bar(x=list(h), y=[v / tot for v in h.values()], marker_color="#1f4e79", name="mạng thật"))
            fig.add_vline(x=s["avg_shortest_path"], line_dash="dash", line_color="#c0392b",
                          annotation_text=f"L = {s['avg_shortest_path']}")
            fig.add_vline(x=s["random_ER"]["avg_shortest_path"], line_dash="dot", line_color="#7f8c8d",
                          annotation_text=f"L ngẫu nhiên = {s['random_ER']['avg_shortest_path']}", annotation_position="bottom right")
            fig.update_layout(title=s["network"], height=340, margin=dict(l=0, r=0, t=40, b=0),
                              xaxis_title="số bước", yaxis_title="tỉ lệ cặp đỉnh")
            col.plotly_chart(fig, width="stretch")
        df = pd.DataFrame([{"Mạng": s["network"], "Đỉnh (LCC)": s["lcc_nodes"], "Bậc TB": s["avg_degree"],
                            "L thật": s["avg_shortest_path"], "L ngẫu nhiên": s["random_ER"]["avg_shortest_path"],
                            "ln n / ln k": s["log_n_over_log_k"], "C thật": s["avg_clustering"],
                            "C ngẫu nhiên": s["random_ER"]["avg_clustering"], "σ": s["sigma"], "Đường kính": s["diameter"]}
                           for s in SNA.get("small_world", [])])
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("σ = (C/C_rand)/(L/L_rand) > 1 ⇒ thế giới nhỏ: khoảng cách ngắn như mạng ngẫu nhiên nhưng phân cụm cao hơn nhiều.")
    with t[1]:
        a, b = st.columns(2)
        r = SNA.get("ranking", {})
        for col, key_, title, color in [(a, "pagerank_scientists", "PageRank – nhà khoa học", "#6c3483"),
                                        (b, "betweenness_scientists", "Betweenness – nhà khoa học cầu nối", "#117a65")]:
            d = pd.DataFrame(r.get(key_, [])[:15], columns=["Tên", "Điểm"]).iloc[::-1]
            fig = px.bar(d, x="Điểm", y="Tên", orientation="h", color_discrete_sequence=[color], height=460, title=title)
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), yaxis_title="")
            col.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("**Viện có PageRank cao nhất**")
        c1.dataframe(pd.DataFrame(r.get("pagerank_institutions", [])[:10], columns=["Viện", "PageRank"]), hide_index=True, width="stretch")
        c2.markdown("**Giải thưởng có PageRank cao nhất**")
        c2.dataframe(pd.DataFrame(r.get("pagerank_awards", [])[:10], columns=["Giải", "PageRank"]), hide_index=True, width="stretch")
        st.caption(f"Tương quan Spearman giữa PageRank và betweenness (nhà khoa học): {r.get('spearman_pagerank_vs_betweenness_scientists')}")
    with t[2]:
        cm = kg.communities[:15]
        df = pd.DataFrame([{"ID": c["id"], "Kích thước": c["size"], "Lĩnh vực": ", ".join(c["top_fields"][:3]),
                            "Viện": ", ".join(c["top_institutions"][:2]),
                            "Trung tâm": ", ".join(c["top_scientists_pagerank"][:4])} for c in cm])
        fig = px.treemap(df, path=["Lĩnh vực"], values="Kích thước", color="Kích thước", color_continuous_scale="Teal", height=380)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df, hide_index=True, width="stretch")
        cid = st.selectbox("Xem hạt nhân của cộng đồng", [c["id"] for c in cm])
        c = next(x for x in kg.communities if x["id"] == cid)
        st.info(kg.community_summaries.get(str(cid), ""))
        show_html(render(kg, list(dict.fromkeys(c["nucleus"] + c["top_scientists_pagerank"])), height=430), height=450)
    with t[3]:
        kc = SNA.get("k_core", {})
        sh = {int(a): b for a, b in kc.get("shell_sizes", {}).items()}
        fig = px.bar(x=list(sh), y=list(sh.values()), log_y=True, labels={"x": "k", "y": "số đỉnh (log)"},
                     color_discrete_sequence=["#5d8c3e"], height=300, title="Phân rã k-core")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"**Hạt nhân k = {kc.get('k_max')}**: {kc.get('main_core_size')} đỉnh – " +
                    ", ".join(f"{k_}: {v}" for k_, v in kc.get("main_core_by_label", {}).items()))
        st.markdown("**Nhà khoa học trong hạt nhân (theo PageRank):** " + ", ".join(kc.get("main_core_scientists", [])[:30]))
        core = kc.get("main_core_scientists", [])[:25] + kc.get("main_core_other", [])[:10]
        show_html(render(kg, core, height=480), height=500)
    with t[4]:
        q = QUAL
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, q.get("junk_nodes", {}).get("count", "-"), "Nút rác đã loại", f"{q.get('junk_nodes', {}).get('edges_removed', '-')} cạnh")
        kpi(c2, f"{100 * (q.get('temporal', {}).get('consistency_rate') or 0):.1f}%", "Nhất quán thời gian", "thầy sinh trước trò")
        kpi(c3, q.get("advisor_cycles", {}).get("count", "-"), "Chu trình thầy–trò", "phải bằng 0")
        kpi(c4, q.get("enrichment", {}).get("added", "-"), "Cạnh làm giàu", "NER + RE từ văn bản")
        st.write("")
        ner = load_json("ner_eval.json").get("C_hybrid", {})
        pr = q.get("precision", {})
        st.markdown("**Mô hình làm giàu dữ liệu**")
        st.dataframe(pd.DataFrame([
            {"Thành phần": "NER + liên kết thực thể (hybrid)", "Chỉ số": "F1 end-to-end", "Giá trị": ner.get("f1_end_to_end")},
            {"Thành phần": "NER + liên kết thực thể (hybrid)", "Chỉ số": "Recall phát hiện", "Giá trị": ner.get("detection_recall_relaxed")},
            {"Thành phần": "RE luật vai trò + NER", "Chỉ số": "Precision (mẫu)", "Giá trị": (pr.get("enrichment_rule+role+ner") or [None])[0]},
            {"Thành phần": "RE giám sát xa (LogReg)", "Chỉ số": "Precision (mẫu)", "Giá trị": (pr.get("enrichment_ds-logreg") or [None])[0]},
        ]), hide_index=True, width="stretch")
        if QAE:
            st.markdown("**Đánh giá hỏi đáp nhiều bước (bộ 144 câu hỏi)**")
            rows = []
            for cfg, d in QAE.items():
                for grp, v in d.items():
                    if isinstance(v, dict):
                        rows.append({"Cấu hình": cfg, "Nhóm": grp, **v})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
