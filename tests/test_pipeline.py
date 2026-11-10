"""Kiểm thử offline cho bước tiền xử lý + trích xuất (không cần mạng)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util

from scinet.wikitext import parse_infobox, sections_with_links
from scinet.relation_extractor import RelationExtractor, RoleExtractor

WT = (Path(__file__).parent / "fixtures" / "sample_page.wiki").read_text()
spec = importlib.util.spec_from_file_location("ex", Path(__file__).parent.parent / "scripts" / "02_extract.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)


def test_infobox():
    name, box = parse_infobox(WT)
    assert name == "infobox scientist"
    assert box["doctoral_students"]["links"] == ["Carol Student", "Dan Student"]
    assert "Max Planck Society" in box["workplaces"]["links"]
    _, edges = ex.infobox_edges({"title": "Alice Example", "wikitext": WT})
    got = {(e["source"], e["rel_type"], e["target"]) for e in edges}
    assert ("Bob Mentor", "DOCTORAL_ADVISOR_OF", "Alice Example") in got
    assert ("Alice Example", "DOCTORAL_ADVISOR_OF", "Carol Student") in got
    assert ("Erwin Influencer", "INFLUENCED", "Alice Example") in got
    assert ("Alice Example", "WORKS_IN_FIELD", "Theoretical physics") in got
    assert ("Alice Example", "RECEIVED_AWARD", "Nobel Prize in Physics") in got


def test_nlp():
    secs = sections_with_links(WT)
    for s in secs:
        for l in s.links:
            assert s.text[l.start:l.end] == l.surface, (s.text[l.start:l.end], l.surface)
    types = {t: "Scientist" for t in ["Erwin Influencer", "Frank Partner", "Gina Rival",
                                      "Henry Skeptic", "Ivan Claimant", "Jack Neutral"]}
    types.update({"CERN": "Institution", "International Atomic Energy Agency": "Institution"})
    rels = RoleExtractor().extract("Alice Example", secs, types)
    got = {(r.source, r.rel_type, r.target) for r in rels}
    for g in sorted(got):
        print(g)
    assert ("Alice Example", "BUILT_UPON", "Erwin Influencer") in got
    assert ("Alice Example", "CRITICIZED", "Gina Rival") in got
    assert ("Henry Skeptic", "REFUTED", "Alice Example") in got
    assert ("Alice Example", "PRIORITY_DISPUTE", "Ivan Claimant") in got
    assert ("Alice Example", "COLLABORATED_WITH", "Frank Partner") in got
    assert ("Alice Example", "ADVISOR_TO", "International Atomic Energy Agency") in got
    assert not any(t == "Jack Neutral" for _, _, t in got)   # câu phủ định bị loại


if __name__ == "__main__":
    test_infobox(); test_nlp(); print("OK")
