from image_curator.calibration import LabeledVector, calibrate_thresholds, evaluate_thresholds
from image_curator.classification import ClassificationConfig


def test_calibration_reports_coverage_accuracy_unknown_false_accept_and_confusion():
    references = {"alpha": [[1, 0]], "beta": [[0, 1]]}
    records = [
        LabeledVector("alpha", (1, 0), "known"),
        LabeledVector(None, (0.95, 0.05), "unknown-near-alpha"),
        LabeledVector(None, (-1, 0), "unknown-far"),
    ]

    metrics = evaluate_thresholds(records, references,
                                  config=ClassificationConfig(min_similarity=0.8, min_margin=0.05))

    assert metrics.coverage == 2 / 3
    assert metrics.accepted_accuracy == 0.5
    assert metrics.unknown_false_accept == 1
    assert metrics.unknown_false_accept_rate == 0.5
    assert metrics.confusion["unknown"] == {"alpha": 1, "unknown": 1}


def test_calibration_compares_threshold_candidates_from_explicit_labels_only():
    report = calibrate_thresholds(
        [LabeledVector("alpha", (1, 0)), LabeledVector(None, (0.95, 0.05))],
        {"alpha": [[1, 0]], "beta": [[0, 1]]}, similarity_thresholds=[0.8, 0.99], margin_thresholds=[0.0],
    )

    assert report["label_source"].startswith("explicit validation JSON")
    assert len(report["candidates"]) == 2
    assert report["recommended"] is None
    assert report["target_gates"]["min_accepted"] == 5


def test_calibration_recommends_only_after_accuracy_safety_and_sample_gates_pass():
    report = calibrate_thresholds(
        [LabeledVector("alpha", (1, 0), str(index)) for index in range(5)]
        + [LabeledVector(None, (-1, 0), "unknown")],
        {"alpha": [[1, 0]], "beta": [[0, 1]]}, similarity_thresholds=[0.8], margin_thresholds=[0.05],
        min_accepted_accuracy=0.9, max_unknown_false_accept_rate=0.0, min_accepted=5, min_unknown=1,
    )

    assert report["recommendation_eligible"] is True
    assert report["recommended"]["accepted_accuracy"] == 1.0
