import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import Checkpoint, CheckpointStore


def test_checkpoint_data():
    mgr = CheckpointStore()
    assert mgr._list is None
    assert mgr._main_path == "checkpoints/checkpoints.yaml"
    checkpoints = mgr.list()
    assert isinstance(checkpoints, list) and len(checkpoints) >= 1
    assert all(isinstance(c, Checkpoint) for c in checkpoints)
    # First checkpoint: checkpoint_01
    c1 = checkpoints[0]
    assert c1.id == "checkpoint_01"
    assert c1.path == "checkpoint_01.pt"
    assert c1.full_path == Path("checkpoints/checkpoint_01.pt")
    assert c1.gdrive_id == "11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv"
    assert len(c1.info) > 0
    assert isinstance(c1.exists, bool)
    # find by 1-based int, string digit, and id/path
    found_by_int = mgr.find(1)
    found_by_str_digit = mgr.find("1")
    found_by_id = mgr.find("checkpoint_01")
    found_by_full_path = mgr.find("checkpoints/checkpoint_01.pt")
    assert found_by_int is c1
    assert found_by_str_digit is c1
    assert found_by_id is c1
    assert found_by_full_path is c1
    # Test auto-discovery of extra *.pt files in base_dir
    dummy_pt = Path("checkpoints/dummy_extra_test.pt")
    try:
        dummy_pt.touch()
        mgr.clear()
        discovered = mgr.find("dummy_extra_test.pt")
        assert discovered is not None
        assert discovered.id == "dummy_extra_test.pt"
        assert discovered.path == "dummy_extra_test.pt"
        assert discovered.full_path == Path("checkpoints/dummy_extra_test.pt")
    finally:
        if dummy_pt.exists():
            dummy_pt.unlink()
        mgr.clear()
    # Test download via mocked gdown when file is missing
    mock_gdown = MagicMock()
    missing_cp = Checkpoint(
        id="mock_remote.pt",
        path="mock_remote.pt",
        gdrive_id="mock_gdrive_123",
        base="checkpoints",
    )
    with patch.dict("sys.modules", {"gdown": mock_gdown}):
        # Mock gdown creating the file
        def fake_download(id, output, quiet, resume=False):
            Path(output).touch()
        mock_gdown.download = MagicMock(side_effect=fake_download)
        out_path = missing_cp.download(quiet=True)
        assert isinstance(out_path, Path)
        assert out_path.exists()
        assert out_path.name == "mock_remote.pt"
        mock_gdown.download.assert_called_once_with(
            id="mock_gdrive_123", output=str(Path(missing_cp.full_path).resolve()), quiet=True, resume=True
        )
        if Path(out_path).exists():
            Path(out_path).unlink()
    # Negative lookup cases
    assert mgr.find(999) is None
    assert mgr.find("0") is None
    assert mgr.find(-1) is None
    assert mgr.find("nonexistent_checkpoint_file.pt") is None
    # Test Checkpoint.parse
    cp_test = Checkpoint(id="custom", path="")
    cp_test.parse(
        {"path": "custom_model.pt", "info": "Custom model", "gdrive_id": "test_id"},
        Path("checkpoints"),
        "custom",
    )
    assert cp_test.path == "custom_model.pt"
    assert cp_test.base == "checkpoints"
    assert cp_test.info == "Custom model"
    assert cp_test.gdrive_id == "test_id"
    assert cp_test.full_path == Path("checkpoints/custom_model.pt")
    # Test Checkpoint.parse with absolute path inside checkpoints directory
    abs_path_test = (Path("checkpoints").resolve() / "abs_model.pt").as_posix()
    cp_abs = Checkpoint(id="abs_model", path="")
    cp_abs.parse(
        {"path": abs_path_test, "info": "Absolute model"},
        Path("checkpoints"),
        "abs_model",
    )
    assert cp_abs.path == "abs_model.pt"
    assert cp_abs.full_path == Path("checkpoints/abs_model.pt")
    mgr.clear()
    assert mgr._list is None
    print("[PASS] CheckpointData (load, list, find, clear) verified.")


if __name__ == "__main__":
    test_checkpoint_data()
    print("\n🎉 ALL TESTS IN tests/test_checkpoint.py PASSED SUCCESSFULLY!")
