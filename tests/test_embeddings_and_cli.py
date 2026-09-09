import json

import pytest

from image_curator.cli import main
from image_curator.embeddings import moat_float16_blob, moat_vector_from_blob


def test_moat_vector_is_normalised_and_round_trips_as_float16():
    blob = moat_float16_blob([3.0, 4.0], dimensions=2)
    vector = moat_vector_from_blob(blob, dimensions=2)

    assert len(blob) == 4
    assert sum(value * value for value in vector) == pytest.approx(1.0, abs=0.001)
    assert vector[0] == pytest.approx(0.6, abs=0.001)


def test_cli_initializes_and_reads_local_checkpoint(tmp_path, capsys):
    database = tmp_path / "state.sqlite"

    assert main(["init-db", str(database)]) == 0
    assert json.loads(capsys.readouterr().out)["created"] is True
    assert main(["status", str(database)]) == 0
    assert json.loads(capsys.readouterr().out)["states"] == {}


def test_cli_scans_and_classifies_with_user_provided_json(tmp_path, capsys):
    library = tmp_path / "library"
    library.mkdir()
    (library / "image.jpg").write_bytes(b"pixels")
    database = tmp_path / "output" / "checkpoint.sqlite"
    references = tmp_path / "references.json"
    vector = tmp_path / "vector.json"
    references.write_text(json.dumps({"known": [[1, 0]], "other": [[0, 1]]}), encoding="utf-8")
    vector.write_text(json.dumps({"vector": [1, 0]}), encoding="utf-8")

    assert main(["scan", str(database), str(library)]) == 0
    scan = json.loads(capsys.readouterr().out)
    assert scan["enqueued"] == 1
    assert scan["read_only_sources"] is True
    assert main(["classify", "--references", str(references), "--vector", str(vector)]) == 0
    classification = json.loads(capsys.readouterr().out)
    assert classification["classification"]["label"] == "known"
    assert classification["route"]["stage"] == "reference"


def test_cli_doctor_makes_user_supplied_weights_boundary_explicit(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["weights_bundled"] is False
    assert "InferenceAdapter" in report["inference"]
