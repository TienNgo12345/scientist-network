"""Bước 3 – Hợp nhất, chuẩn hoá thực thể, lọc và xuất đồ thị.

Đầu ra: data/output/nodes.csv, data/output/edges.csv
  nodes.csv : id, name, label, url, crawled
  edges.csv : source, target, type, method, confidence, evidence, section, page
"""
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jellyfish
import pandas as pd

from scinet.config import RAW, INTERIM, OUT
from scinet.wiki_client import read_jsonl

INST_RX = re.compile(
    r"(universit|college|institut|laborator|school|academy|society|hospital|"
    r"cent(er|re)|observatory|foundation|polytechn|ETH |CERN|council|agency|"
    r"organi[sz]ation|corporation|labs?\b|museum|NASA|IBM|Bell Labs|Max Planck)", re.I)
SYMMETRIC = {"COLLABORATED_WITH", "PRIORITY_DISPUTE"}
STOP_PLAIN = {"see below", "list", "various", "none", "n/a", "others", "et al"}


def canon(t, aliases):
    for _ in range(3):
        t = aliases.get(t, t)
    return t


def key_of(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def cluster_names(names, threshold=0.96):
    """Gộp các tên gần trùng (Jaro-Winkler) cho nút Field/Award dạng văn bản."""
    rep = {}
    reps = []
    for n in sorted(names, key=lambda x: (-len(x), x)):
        k = key_of(n)
        hit = None
        for r in reps:
            if k == key_of(r) or jellyfish.jaro_winkler_similarity(k, key_of(r)) >= threshold:
                hit = r
                break
        if hit is None:
            reps.append(n)
            hit = n
        rep[n] = hit
    return rep


def main():
    crawled = {r["title"] for r in read_jsonl(RAW / "scientists.jsonl")}
    link_types = json.loads((INTERIM / "link_types.json").read_text(encoding="utf-8"))
    ap = INTERIM / "aliases.json"
    aliases = json.loads(ap.read_text(encoding="utf-8")) if ap.exists() else {}
    ib = list(read_jsonl(INTERIM / "infobox_edges.jsonl"))
    nl = list(read_jsonl(INTERIM / "nlp_edges.jsonl"))

    labels = {}           # node name -> label
    for t in crawled:
        labels[t] = "Scientist"

    def typed(name, expected):
        """Trả về tên chuẩn nếu nút đúng loại mong đợi, ngược lại None."""
        c = canon(name, aliases)
        kind = link_types.get(c) or link_types.get(name)
        if expected == "Scientist":
            return c if (c in crawled or kind == "Scientist") else None
        if expected == "Institution":
            return c if (kind == "Institution" or INST_RX.search(c)) and kind != "Scientist" else None
        return c

    edges = []
    plain_by_type = defaultdict(set)

    def valid_plain(o, tt):
        """Lọc mảnh văn bản rác còn lại sau khi bỏ link (vd 'in 1978', 'Fellow of the  since 2016')."""
        if "  " in o or len(o) < 4 or len(o) > 70:
            return False
        low = o.lower()
        if low.split()[0] in ("in", "of", "and", "for", "the", "from", "with", "at", "by", "on", "to", "since", "list"):
            return False
        if low.split()[-1] in ("of", "the", "and", "in", "for", "to"):
            return False
        if re.fullmatch(r"[\d\W]+", o) or re.search(r"\b(1[5-9]|20)\d\d\b", o) and tt == "Field":
            return False
        return key_of(o) not in STOP_PLAIN

    # Bỏ năm ở cuối tên giải: "Biophysical Society Innovation Award 2021" -> giữ nguyên tên giải
    resolved = []
    votes = defaultdict(lambda: defaultdict(int))
    for e in ib:
        tt = e["target_type"]
        other = e["target"] if e["source"] == e["page"] else e["source"]
        page = canon(e["page"], aliases)
        if e["is_link"]:
            if other.startswith(("List of", "Lists of")):
                continue
            if tt in ("Scientist", "Institution"):
                o = typed(other, tt)
            else:
                o = canon(other, aliases)
                kind = link_types.get(o) or link_types.get(other)
                if kind in ("Scientist", "Institution"):   # vd "Fellow of the [[Royal Society of Edinburgh]]" trong awards
                    continue
        else:
            o = re.sub(r"\s*\(?\b(1[5-9]|20)\d\d\)?\s*$", "", other.strip()).strip(" ,;:")
            if not valid_plain(o, tt):
                continue
            o = o[0].upper() + o[1:]
            plain_by_type[tt].add(o)
        if not o:
            continue
        votes[o][tt] += 1
        resolved.append((e, tt, o, page))

    for o, v in votes.items():
        if o in labels:
            continue
        kind = link_types.get(o)
        labels[o] = kind if kind in ("Scientist", "Institution") else max(v.items(), key=lambda x: x[1])[0]

    for e, tt, o, page in resolved:
        if labels.get(o) != tt:          # nút đã được gán loại khác theo đa số -> bỏ cạnh mâu thuẫn
            continue
        src, dst = (page, o) if e["source"] == e["page"] else (o, page)
        edges.append(dict(source=src, target=dst, type=e["rel_type"], method="infobox",
                          confidence=e["confidence"], evidence=f"infobox:{e['field']}",
                          section="Infobox", page=page))

    for e in nl:
        tt = "Institution" if e["rel_type"] == "ADVISOR_TO" else "Scientist"
        s = canon(e["source"], aliases)
        t = canon(e["target"], aliases)
        page = canon(e["page"], aliases)
        if tt == "Institution":
            if typed(t, "Institution") is None:
                continue
            labels.setdefault(t, "Institution")
        else:
            other = t if s == page else s
            if typed(other, "Scientist") is None:
                continue
            labels.setdefault(other, "Scientist")
        edges.append(dict(source=s, target=t, type=e["rel_type"], method=e["method"],
                          confidence=e["confidence"], evidence=e["evidence"],
                          section=e["section"], page=page))

    # Gộp tên Field/Award dạng text bằng Jaro-Winkler
    rep = {}
    for tt, names in plain_by_type.items():
        rep.update(cluster_names(names))
    for n_, r_ in rep.items():
        if n_ in labels:
            labels.setdefault(r_, labels[n_])
    for e in edges:
        e["source"] = rep.get(e["source"], e["source"])
        e["target"] = rep.get(e["target"], e["target"])

    df = pd.DataFrame(edges)
    df = df[df.source != df.target]
    sym = df.type.isin(SYMMETRIC)
    a = df.loc[sym, ["source", "target"]].min(axis=1)
    b = df.loc[sym, ["source", "target"]].max(axis=1)
    df.loc[sym, "source"], df.loc[sym, "target"] = a, b
    df = (df.sort_values("confidence", ascending=False)
            .drop_duplicates(["source", "target", "type"]))

    used = set(df.source) | set(df.target)
    nodes = pd.DataFrame([{"name": n, "label": labels.get(n, "Scientist"),
                           "crawled": n in crawled} for n in sorted(used)])
    nodes.insert(0, "id", range(len(nodes)))
    nodes["url"] = nodes.apply(
        lambda r: "https://en.wikipedia.org/wiki/" + r["name"].replace(" ", "_")
        if r["label"] in ("Scientist", "Institution") or r["crawled"] else "", axis=1)

    nodes.to_csv(OUT / "nodes.csv", index=False)
    df.to_csv(OUT / "edges.csv", index=False)
    print(nodes.label.value_counts().to_string())
    print(df.type.value_counts().to_string())
    print(f"Tổng: {len(nodes)} nút, {len(df)} cạnh -> {OUT}")


if __name__ == "__main__":
    main()
