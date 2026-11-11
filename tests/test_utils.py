import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

import numpy as np
import torch

from config import AttributeDict, utils
from config.utils import _soundfile_load, _soundfile_save


def test_attribute_dict():
    # 1. Initialization from dict
    d = AttributeDict({"a": 1, "b": "hello", "c": [1, 2, 3]})
    assert d.a == 1
    assert d["a"] == 1
    assert d.b == "hello"
    assert d.c == [1, 2, 3]

    # 2. Attribute setting
    d.new_attr = 42
    assert d.new_attr == 42
    assert d["new_attr"] == 42

    # 3. Attribute deletion
    del d.new_attr
    assert "new_attr" not in d
    try:
        _ = d.new_attr
        assert False, "Should have raised AttributeError for deleted key"
    except AttributeError:
        pass

    try:
        del d.non_existent_key
        assert False, "Should have raised AttributeError when deleting nonexistent key"
    except AttributeError:
        pass

    # 4. Missing attribute access
    try:
        _ = d.non_existent
        assert False, "Should have raised AttributeError on missing attribute"
    except AttributeError:
        pass

    # 5. Standard dict operations
    assert "a" in d
    assert list(d.keys()) == ["a", "b", "c"]
    assert d.get("a") == 1
    assert d.get("missing", "default") == "default"

    print("[PASS] AttributeDict attributes, dict access, and errors verified.")


def test_soundfile_fallbacks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav_path = tmp_path / "test.wav"

        # 1. _soundfile_save with 1D tensor
        sr = 24000
        tensor_1d = torch.zeros((sr,))
        _soundfile_save(str(wav_path), tensor_1d, sr)
        assert wav_path.exists()

        # 2. _soundfile_load
        loaded_tensor, loaded_sr = _soundfile_load(str(wav_path))
        assert loaded_sr == sr
        assert loaded_tensor is not None
        assert hasattr(loaded_tensor, "ndim") or hasattr(loaded_tensor, "shape")
        assert loaded_tensor.ndim == 2  # (1, num_samples)

        # 3. Auto-appending .wav extension if missing
        no_ext_path = tmp_path / "test_no_ext"
        _soundfile_save(str(no_ext_path), tensor_1d, sr)
        expected_wav = tmp_path / "test_no_ext.wav"
        assert expected_wav.exists()

    print("[PASS] Soundfile IO fallbacks and extension handling verified.")


if __name__ == "__main__":
    test_attribute_dict()
    test_soundfile_fallbacks()
    print("\n🎉 ALL TESTS IN tests/test_utils.py PASSED SUCCESSFULLY!")
