import json
from pathlib import Path
import numpy as np
from influence_curriculum.data import DataConfig, load_documents
from influence_curriculum.curriculum import CurriculumConfig, build_curriculum


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


def test_mean_gradient_identity():
    """dot-with-mean == (1/D)*sum_of_pairwise for any unit-normalized gradient matrix."""
    rng = np.random.default_rng(0)
    D, V = 8, 50
    raw = rng.random((D, V)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    grads = raw / norms  # unit-normalized

    efficient = grads @ grads.mean(axis=0)                              # O(D)
    pairwise = np.array([(1 / D) * (grads[i] @ grads.T).sum()          # O(D^2)
                         for i in range(D)])
    np.testing.assert_allclose(efficient, pairwise, rtol=1e-5)


def test_permutation_validity(tmp_path):
    D, T = 20, 3
    rng = np.random.default_rng(42)
    Phi = rng.random((D, T)).astype(np.float32)
    texts = [f"doc {i}" for i in range(D)]
    doc_ids = [f"fake#{i}" for i in range(D)]
    build_curriculum(Phi, texts, doc_ids, CurriculumConfig(segment_size=5), str(tmp_path), seed=0)
    for e in range(T):
        path = tmp_path / "curriculum" / f"epoch_{e:02d}.jsonl"
        assert path.exists()
        ids = [json.loads(l)["id"] for l in path.read_text().splitlines()]
        assert sorted(ids) == sorted(doc_ids), f"epoch {e} is not a permutation"


def test_determinism(tmp_path):
    D, T = 10, 2
    rng = np.random.default_rng(7)
    Phi = rng.random((D, T)).astype(np.float32)
    texts = [f"doc {i}" for i in range(D)]
    doc_ids = [f"fake#{i}" for i in range(D)]
    cfg = CurriculumConfig()
    build_curriculum(Phi, texts, doc_ids, cfg, str(tmp_path / "r1"), seed=42)
    build_curriculum(Phi, texts, doc_ids, cfg, str(tmp_path / "r2"), seed=42)
    for e in range(T):
        f1 = (tmp_path / "r1" / "curriculum" / f"epoch_{e:02d}.jsonl").read_text()
        f2 = (tmp_path / "r2" / "curriculum" / f"epoch_{e:02d}.jsonl").read_text()
        assert f1 == f2, f"epoch {e} differs between identical runs"
