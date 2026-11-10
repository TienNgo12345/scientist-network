"""Phân tích wikitext: Infobox, chuyển sang văn bản sạch có giữ vị trí liên kết.

Đây là bước tiền xử lý quan trọng nhất: thay vì dùng NER "mù" trên văn bản thuần,
ta giữ lại các wikilink [[Trang|nhãn]] cùng vị trí ký tự của chúng. Mỗi wikilink
là một thực thể ĐÃ được liên kết (entity linking) tới đúng trang Wikipedia,
nên độ chính xác của bước nhận dạng + liên kết thực thể gần như tuyệt đối.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mwparserfromhell as mwp

from .config import SECTION_PATTERN, SCIENTIST_INFOBOXES, INSTITUTION_INFOBOXES

_SECTION_RE = re.compile(SECTION_PATTERN, re.I)
_SKIP_LINK_NS = ("file:", "image:", "category:", "wikt:", "wiktionary:", "s:",
                 "commons:", "help:", "wikipedia:", "template:", "portal:", ":category:")


def norm_title(t: str) -> Optional[str]:
    """Chuẩn hoá tên trang từ wikilink: bỏ #anchor, viết hoa chữ đầu, NFC."""
    t = unicodedata.normalize("NFC", t).strip().replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    if not t or t.lower().startswith(_SKIP_LINK_NS):
        return None
    t = t.split("#")[0].strip()
    if not t:
        return None
    return t[0].upper() + t[1:]


def normalize_text(s: str) -> str:
    """Chuẩn hoá văn bản: Unicode NFC, khoảng trắng, dấu nháy."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("’", "'").replace("‘", "'").replace(" ", " ")
    s = re.sub(r"'{2,}", "", s)                 # ''italic'' / '''bold'''
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Infobox
# ---------------------------------------------------------------------------
def infobox_name(code) -> Optional[str]:
    for tpl in code.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower().replace("_", " ")
        if name.startswith("infobox"):
            return re.sub(r"\s+", " ", name)
    return None


def classify_page(wikitext: str) -> str:
    """Scientist / Institution / Other dựa trên template Infobox."""
    code = mwp.parse(wikitext)
    name = infobox_name(code)
    if not name:
        return "Other"
    if name in SCIENTIST_INFOBOXES:
        return "Scientist"
    if name in INSTITUTION_INFOBOXES:
        return "Institution"
    return "Other"


def _plain_items(value) -> List[str]:
    """Các mục KHÔNG có link trong 1 trường (vd: 'Physics, Chemistry')."""
    txt = value.strip_code(normalize=True, collapse=True)
    txt = re.sub(r"\(.*?\)", "", txt)
    parts = re.split(r"[\n,;•]|\band\b", txt)
    return [p.strip(" .") for p in parts if 2 < len(p.strip(" .")) < 80]


def parse_infobox(wikitext: str) -> Tuple[Optional[str], Dict[str, dict]]:
    """Trả về (tên infobox, {field: {"links": [title...], "plain": [text...]}})."""
    code = mwp.parse(wikitext)
    out: Dict[str, dict] = {}
    for tpl in code.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower().replace("_", " ")
        if not name.startswith("infobox"):
            continue
        for p in tpl.params:
            key = str(p.name).strip().lower().replace(" ", "_")
            links = []
            for wl in p.value.filter_wikilinks():
                t = norm_title(str(wl.title))
                if t:
                    links.append(t)
            # bỏ phần trong <ref> khi lấy plain text
            for ref in p.value.filter_tags(matches=lambda n: n.tag == "ref"):
                try:
                    p.value.remove(ref)
                except ValueError:
                    pass
            # phần văn bản không có link (vd "Theoretical physics, [[Quantum mechanics]]")
            rest = mwp.parse(str(p.value))
            for wl in rest.filter_wikilinks(recursive=False):
                try:
                    rest.remove(wl)
                except ValueError:
                    pass
            out[key] = {"links": list(dict.fromkeys(links)), "plain": _plain_items(rest)}
        return re.sub(r"\s+", " ", name), out
    return None, out


# ---------------------------------------------------------------------------
# Văn bản theo mục, giữ vị trí link
# ---------------------------------------------------------------------------
@dataclass
class LinkSpan:
    start: int
    end: int
    title: str
    surface: str


@dataclass
class Section:
    heading: str
    text: str
    links: List[LinkSpan] = field(default_factory=list)


def _render(nodes, buf: List[str], links: List[LinkSpan], pos: List[int]):
    for n in nodes:
        if isinstance(n, mwp.nodes.Text):
            s = str(n)
            buf.append(s); pos[0] += len(s)
        elif isinstance(n, mwp.nodes.Wikilink):
            t = norm_title(str(n.title))
            if t is None:
                continue                        # File:, Category: ...
            surface = (n.text.strip_code() if n.text else str(n.title)).strip()
            surface = normalize_text(surface)
            if not surface:
                continue
            links.append(LinkSpan(pos[0], pos[0] + len(surface), t, surface))
            buf.append(surface); pos[0] += len(surface)
        elif isinstance(n, mwp.nodes.ExternalLink):
            s = n.title.strip_code() if n.title else ""
            buf.append(s); pos[0] += len(s)
        elif isinstance(n, mwp.nodes.Tag):
            if str(n.tag).lower() in ("ref", "math", "table", "gallery", "sup", "sub", "small", "references"):
                if str(n.tag).lower() in ("sup", "sub", "small") and n.contents:
                    _render(n.contents.nodes, buf, links, pos)
                continue
            if n.contents:
                _render(n.contents.nodes, buf, links, pos)
        elif isinstance(n, mwp.nodes.Template):
            name = str(n.name).strip().lower()
            # một số template mang văn bản cần giữ lại
            if name in ("lang", "nowrap", "small") and n.params:
                _render(n.params[-1].value.nodes, buf, links, pos)
            elif name in ("ill", "interlanguage link") and n.params:
                t = norm_title(str(n.params[0].value))
                s = n.params[0].value.strip_code().strip()
                if t and s:
                    links.append(LinkSpan(pos[0], pos[0] + len(s), t, s))
                    buf.append(s); pos[0] += len(s)
            # còn lại (cite, efn, sfn, infobox...) bỏ qua
        elif isinstance(n, (mwp.nodes.Comment, mwp.nodes.Heading)):
            continue
        else:
            s = str(n.__strip__(normalize=True, collapse=True) or "")
            buf.append(s); pos[0] += len(s)


def _clean_with_links(raw: str, links: List[LinkSpan]) -> Tuple[str, List[LinkSpan]]:
    """Chuẩn hoá khoảng trắng nhưng vẫn ánh xạ lại offset của link."""
    keep, mapping = [], []
    prev_space = False
    for i, ch in enumerate(raw):
        if ch in " \t ":
            if prev_space:
                mapping.append(len(keep)); continue
            ch = " "; prev_space = True
        else:
            prev_space = False
        mapping.append(len(keep)); keep.append(ch)
    mapping.append(len(keep))
    text = "".join(keep)
    new_links = [LinkSpan(mapping[l.start], mapping[l.end], l.title, l.surface) for l in links]
    return text, new_links


def sections_with_links(wikitext: str, only_relevant: bool = True) -> List[Section]:
    code = mwp.parse(wikitext)
    result = []
    for sec in code.get_sections(levels=[2, 3], include_lead=True, flat=True):
        heads = sec.filter_headings(recursive=False)
        heading = heads[0].title.strip_code().strip() if heads else "Lead"
        if heading.lower() in ("references", "notes", "external links", "see also",
                               "bibliography", "further reading", "publications",
                               "selected publications", "honours and awards", "awards"):
            continue
        if only_relevant and heading != "Lead" and not _SECTION_RE.search(heading):
            continue
        buf, links, pos = [], [], [0]
        _render(sec.nodes, buf, links, pos)
        text, links = _clean_with_links("".join(buf), links)
        if text.strip():
            result.append(Section(heading, text, links))
    return result


def all_links(wikitext: str) -> List[str]:
    code = mwp.parse(wikitext)
    out = []
    for wl in code.filter_wikilinks():
        t = norm_title(str(wl.title))
        if t:
            out.append(t)
    return list(dict.fromkeys(out))
