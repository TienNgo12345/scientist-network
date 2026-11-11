"""Client tối giản cho MediaWiki API (en.wikipedia.org) có cache và giới hạn tốc độ."""
import json
import time
from typing import Dict, Iterable, List

import requests

from .config import WIKI_API, USER_AGENT, REQUEST_DELAY, BATCH


class WikiClient:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self._last = 0.0

    def _get(self, params: dict) -> dict:
        """Gọi API; dùng POST (tránh URL quá dài), thử lại tối đa 6 lần, báo rõ lỗi cuối."""
        params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
        last_err = ""
        for attempt in range(6):
            wait = REQUEST_DELAY - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.post(WIKI_API, data=params, timeout=60)
                if r.status_code in (429, 503):
                    last_err = f"HTTP {r.status_code} (bị giới hạn tốc độ)"
                    time.sleep(int(r.headers.get("Retry-After", 5 * (attempt + 1))))
                    continue
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(3 * (attempt + 1))
                    continue
                data = r.json()
                if "error" in data:
                    code = data["error"].get("code")
                    last_err = f"API error {code}: {data['error'].get('info')}"
                    if code == "maxlag":
                        time.sleep(5)
                        continue
                    break
                return data
            except (requests.RequestException, ValueError) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(
            f"Không gọi được Wikipedia API sau nhiều lần thử. Lỗi cuối: {last_err}\n"
            "Nếu là HTTP 403 hoặc lỗi kết nối: mạng của bạn đang chặn API. Hãy dùng cách thu thập qua "
            "trình duyệt (scripts/browser_crawler.js, xem README) rồi chạy scripts/00_import_browser_dump.py.")

    # ---- Category ---------------------------------------------------------
    def category_members(self, category: str, limit: int = 2000) -> List[str]:
        out, cont = [], {}
        while len(out) < limit:
            data = self._get({"action": "query", "list": "categorymembers",
                              "cmtitle": category, "cmtype": "page",
                              "cmnamespace": 0, "cmlimit": 500, **cont})
            out += [m["title"] for m in data["query"]["categorymembers"]]
            if "continue" not in data:
                break
            cont = data["continue"]
        return out[:limit]

    # ---- Wikitext theo lô 50 trang, tự giải redirect ------------------------
    def fetch_wikitext(self, titles: Iterable[str]) -> Dict[str, dict]:
        """Trả về {title_yêu_cầu: {"title": canonical, "pageid", "wikitext"}}."""
        titles = list(dict.fromkeys(titles))
        result = {}
        for i in range(0, len(titles), BATCH):
            chunk = titles[i:i + BATCH]
            data = self._get({"action": "query", "prop": "revisions",
                              "rvprop": "content", "rvslots": "main",
                              "redirects": 1, "titles": "|".join(chunk)})
            q = data.get("query", {})
            alias = {}
            for n in q.get("normalized", []):
                alias[n["from"]] = n["to"]
            for rd in q.get("redirects", []):
                alias[rd["from"]] = rd["to"]
            pages = {}
            for p in q.get("pages", []):
                if p.get("missing") or "revisions" not in p:
                    continue
                pages[p["title"]] = {
                    "title": p["title"], "pageid": p["pageid"],
                    "wikitext": p["revisions"][0]["slots"]["main"]["content"],
                }
            for t in chunk:
                c = t
                for _ in range(3):           # normalized -> redirect -> ...
                    c = alias.get(c, c)
                if c in pages:
                    result[t] = pages[c]
        return result

    def resolve_titles(self, titles: Iterable[str]) -> Dict[str, str]:
        """Chuẩn hoá tên trang (redirect/normalize) -> tên canonical. Trang không tồn tại bị bỏ."""
        titles = list(dict.fromkeys(titles))
        out = {}
        for i in range(0, len(titles), BATCH):
            chunk = titles[i:i + BATCH]
            data = self._get({"action": "query", "redirects": 1,
                              "titles": "|".join(chunk)})
            q = data.get("query", {})
            alias = {n["from"]: n["to"] for n in q.get("normalized", [])}
            alias.update({r["from"]: r["to"] for r in q.get("redirects", [])})
            exist = {p["title"] for p in q.get("pages", []) if not p.get("missing")}
            for t in chunk:
                c = t
                for _ in range(3):
                    c = alias.get(c, c)
                if c in exist:
                    out[t] = c
        return out


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _tpl_names(names):
    return "|".join("Template:" + n[0].upper() + n[1:] for n in names)


def classify_titles(client: "WikiClient", titles, scientist_tpls, institution_tpls):
    """Phân loại nhanh trang (không tải nội dung) qua prop=templates.

    Trả về {title_yêu_cầu: (canonical_title, "Scientist"|"Institution"|"Other")}.
    """
    sci = {"Template:" + n[0].upper() + n[1:] for n in scientist_tpls}
    inst = {"Template:" + n[0].upper() + n[1:] for n in institution_tpls}
    filt = "|".join(sorted(sci | inst))
    titles = list(dict.fromkeys(titles))
    out = {}
    for i in range(0, len(titles), BATCH):
        chunk = titles[i:i + BATCH]
        params = {"action": "query", "prop": "templates", "redirects": 1,
                  "tllimit": 500, "tltemplates": filt, "titles": "|".join(chunk)}
        data = client._get(params)
        q = data.get("query", {})
        alias = {n["from"]: n["to"] for n in q.get("normalized", [])}
        alias.update({r["from"]: r["to"] for r in q.get("redirects", [])})
        kinds = {}
        for p in q.get("pages", []):
            if p.get("missing"):
                continue
            tpls = {t["title"] for t in p.get("templates", [])}
            kinds[p["title"]] = ("Scientist" if tpls & sci else
                                 "Institution" if tpls & inst else "Other")
        for t in chunk:
            c = t
            for _ in range(3):
                c = alias.get(c, c)
            if c in kinds:
                out[t] = (c, kinds[c])
    return out
