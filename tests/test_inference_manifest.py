import pytest

from image_curator.inference import require_adapter_manifest


class Manifested:
    name = "example"

    def manifest(self):
        return {
            "name": self.name,
            "version": "1",
            "artifacts": [{"role": "weights", "sha256": "a" * 64}],
        }


def test_versioned_adapter_manifest_is_path_free_and_complete():
    assert require_adapter_manifest(Manifested())["name"] == "example"


def test_versioned_adapter_rejects_missing_manifest_and_paths():
    with pytest.raises(TypeError):
        require_adapter_manifest(object())

    class Leaky(Manifested):
        def manifest(self):
            value = super().manifest()
            value["artifacts"][0]["path"] = "/private/model.onnx"
            return value

    with pytest.raises(ValueError, match="local paths"):
        require_adapter_manifest(Leaky())
