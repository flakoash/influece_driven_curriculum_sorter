import textwrap
from pathlib import Path
from influence_curriculum.data import DataConfig, load_documents


def test_line_segmentation(tmp_path):
    (tmp_path / "src.txt").write_text("hello\nworld\n\nskip me\n")
    texts, ids = load_documents(str(tmp_path), DataConfig(doc_boundary="line", min_doc_tokens=0))
    assert texts == ["hello", "world", "skip me"]
    assert ids == ["src#0", "src#1", "src#2"]


def test_blank_line_segmentation(tmp_path):
    (tmp_path / "src.txt").write_text("para one\nstill one\n\npara two\n")
    texts, ids = load_documents(str(tmp_path), DataConfig(doc_boundary="blank_line", min_doc_tokens=0))
    assert texts == ["para one\nstill one", "para two"]
    assert ids == ["src#0", "src#1"]


def test_doc_id_stable(tmp_path):
    (tmp_path / "a.txt").write_text("x\ny\n")
    (tmp_path / "b.txt").write_text("z\n")
    _, ids = load_documents(str(tmp_path), DataConfig(min_doc_tokens=0))
    assert "a#0" in ids and "a#1" in ids and "b#0" in ids
