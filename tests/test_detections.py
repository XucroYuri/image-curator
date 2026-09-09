from image_curator.detections import classwise_nms, detection_summary


def _row(cx, cy, score, class_index):
    row = [cx, cy, 40, 40] + [0.0] * 18
    row[4 + class_index] = score
    return row


def test_nms_suppresses_only_same_class_overlaps():
    # Class indexes 16 and 3 are covered and exposed breast respectively.
    detections = classwise_nms([_row(100, 100, 0.9, 16), _row(101, 101, 0.8, 16), _row(101, 101, 0.7, 3)])

    assert [(item["class"], item["confidence"]) for item in detections] == [
        ("FEMALE_BREAST_COVERED", 0.9), ("FEMALE_BREAST_EXPOSED", 0.7)
    ]
    assert detection_summary(detections)["explicit_score"] == 0.7
    assert detection_summary(detections)["intimate_covered_score"] == 0.9


def test_transposed_output_is_accepted():
    rows = [_row(100, 100, 0.9, 16), _row(101, 101, 0.8, 16)]
    transposed = [list(column) for column in zip(*rows)]

    assert len(classwise_nms(transposed)) == 1


def test_realistic_wide_transposed_output_is_not_mistaken_for_rows():
    rows = [_row(100, 100, 0.9, 16)] + [_row(0, 0, 0.0, 0) for _ in range(2099)]
    transposed = [[list(column) for column in zip(*rows)]]

    detections = classwise_nms(transposed)

    assert len(detections) == 1
    assert detections[0]["confidence"] == 0.9
