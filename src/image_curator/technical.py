"""Bounded, model-free technical image evidence."""

from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageFilter, ImageStat


def technical_evidence(image: Image.Image, *, max_edge: int = 512) -> dict[str, Any]:
    """Measure inexpensive technical signals on a bounded luminance thumbnail.

    These values are evidence for dataset-specific calibration.  In particular,
    the edge statistic is not a universal blur decision threshold.
    """
    if max_edge < 32:
        raise ValueError("max_edge must be at least 32")
    probe = image.convert("L")
    probe.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    histogram = probe.histogram()
    pixels = max(1, probe.width * probe.height)
    probabilities = (count / pixels for count in histogram if count)
    entropy = -sum(value * math.log2(value) for value in probabilities)
    luminance = ImageStat.Stat(probe)
    edges = probe.filter(ImageFilter.FIND_EDGES)
    if edges.width > 2 and edges.height > 2:
        edges = edges.crop((1, 1, edges.width - 1, edges.height - 1))
    edge_stats = ImageStat.Stat(edges)
    return {
        "probe_width": probe.width,
        "probe_height": probe.height,
        "luminance_mean": round(float(luminance.mean[0]), 4),
        "luminance_stddev": round(float(luminance.stddev[0]), 4),
        "entropy_bits": round(float(entropy), 4),
        "dark_clip_ratio": round(sum(histogram[:3]) / pixels, 6),
        "bright_clip_ratio": round(sum(histogram[253:]) / pixels, 6),
        "edge_mean": round(float(edge_stats.mean[0]), 4),
        "edge_variance_proxy": round(float(edge_stats.var[0]), 4),
        "calibration_required": True,
    }
