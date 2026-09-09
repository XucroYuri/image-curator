import pytest

from image_curator.classification import ClassificationConfig, classify_open_set
from image_curator.inference import CallableInferenceAdapter, InferenceAdapter
from image_curator.routing import route


def test_open_set_accepts_distinct_reference_and_routes_it():
    decision = classify_open_set(
        [0.99, 0.01], {"alpha": [[1, 0], [0.98, 0.02]], "beta": [[0, 1]]},
        config=ClassificationConfig(min_similarity=0.8, min_margin=0.2, top_k=2),
    )

    assert decision.label == "alpha"
    assert decision.state == "accepted"
    assert len(decision.candidates) == 2
    assert route(classification=decision).stage == "reference"


def test_reference_score_combines_class_center_and_top_k_examples():
    decision = classify_open_set(
        [1, 0], {"alpha": [[1, 0], [0, 1]]}, config=ClassificationConfig(top_k=1, min_similarity=0.0),
    )

    assert decision.candidates[0][1] == 0.8535533905932737


def test_open_set_rejects_weak_and_ambiguous_matches_for_review():
    weak = classify_open_set([1, 0], {"alpha": [[0, 1]], "beta": [[0, -1]]})
    ambiguous = classify_open_set(
        [1, 0], {"alpha": [[1, 0]], "beta": [[0.99, 0.01]]},
        config=ClassificationConfig(min_similarity=0.8, min_margin=0.05),
    )

    assert weak.state == "unknown"
    assert ambiguous.state == "unknown"
    assert route(classification=weak).state == "review"
    assert route(rule_accepted=True, rule_reason="metadata rule").stage == "rule"


def test_callable_adapter_has_no_model_runtime_requirement(tmp_path):
    adapter = CallableInferenceAdapter("test", lambda data, path: [len(data), len(path.name)])

    assert isinstance(adapter, InferenceAdapter)
    assert list(adapter.embed(b"abc", tmp_path / "image.jpg")) == [3, 9]


def test_open_set_rejects_empty_reference_class():
    with pytest.raises(ValueError, match="at least one vector"):
        classify_open_set([1, 0], {"empty": []})
