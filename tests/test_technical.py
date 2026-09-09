import pytest
from PIL import Image

from image_curator.technical import technical_evidence


def test_technical_evidence_is_bounded_and_separates_flat_from_detailed_image():
    flat = Image.new("RGB", (1024, 512), "gray")
    checker = Image.new("L", (64, 64))
    checker.putdata([255 if (x + y) % 2 else 0 for y in range(64) for x in range(64)])

    flat_result = technical_evidence(flat, max_edge=128)
    checker_result = technical_evidence(checker, max_edge=128)

    assert flat_result["probe_width"] == 128
    assert flat_result["probe_height"] == 64
    assert flat_result["calibration_required"] is True
    assert checker_result["edge_variance_proxy"] > flat_result["edge_variance_proxy"]


def test_technical_evidence_rejects_unreasonably_small_probe():
    with pytest.raises(ValueError, match="at least 32"):
        technical_evidence(Image.new("RGB", (8, 8)), max_edge=16)
