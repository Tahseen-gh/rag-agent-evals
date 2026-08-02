from pathlib import Path

from ml.reranker import RerankerWeights, train_and_evaluate
from ml.train_loop import main


def test_train_and_evaluate_produces_valid_report(tmp_path: Path) -> None:
    report = train_and_evaluate(seed=7, val_fraction=0.25, weights_path=tmp_path / "weights.json")

    assert 0.0 <= report["train_accuracy"] <= 1.0
    assert 0.0 <= report["val_accuracy"] <= 1.0
    assert report["n_train_queries"] + report["n_val_queries"] > 0
    assert set(report["weights"].keys()) == {"coef_bm25", "coef_embed", "intercept"}
    assert (tmp_path / "weights.json").exists()


def test_train_and_evaluate_is_deterministic(tmp_path: Path) -> None:
    r1 = train_and_evaluate(seed=42, weights_path=tmp_path / "a.json")
    r2 = train_and_evaluate(seed=42, weights_path=tmp_path / "b.json")
    assert r1["weights"] == r2["weights"]
    assert r1["train_accuracy"] == r2["train_accuracy"]


def test_reranker_weights_score_is_a_probability() -> None:
    weights = RerankerWeights(coef_bm25=1.0, coef_embed=1.0, intercept=0.0)
    assert 0.0 <= weights.score(0.0, 0.0) <= 1.0
    assert weights.score(1.0, 1.0) > weights.score(0.0, 0.0)


def test_main_writes_default_artifact() -> None:
    report = main()
    assert "held_out_recall_at_3" in report
