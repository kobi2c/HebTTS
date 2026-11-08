import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import (
    Config,
    MbdMode,
    RandomMode,
    NikudMode,
    DEFAULT_TEXT,
)


def test_enums():
    for enum_cls in (MbdMode, RandomMode, NikudMode):
        assert enum_cls.Disabled == 0
        assert enum_cls.Enabled == 1
        assert int(enum_cls.Disabled) == 0
        assert int(enum_cls.Enabled) == 1
        assert bool(enum_cls.Enabled) is True
        assert bool(enum_cls.Disabled) is False
    print("[PASS] Enums verified.")


def test_config_defaults():
    cfg = Config()
    assert cfg.checkpoint == "checkpoint.pt"
    assert cfg.speaker_mode == "Predefined Speaker"
    assert cfg.predefined_speaker in ("osim", "1")
    assert cfg.clone_audio is None
    assert cfg.clone_text == ""
    assert cfg.text == DEFAULT_TEXT
    assert cfg.pause_duration == 0.3
    assert cfg.temperature == 1.0
    assert cfg.top_k == 50
    assert cfg.mbd == MbdMode.Disabled
    assert cfg.seed_number == 420
    assert cfg.randomize_seed == RandomMode.Enabled
    assert cfg.nikud_converter == NikudMode.Enabled
    print("[PASS] Config defaults verified.")


def test_config_dicts():
    cfg = Config(temperature=1.2, top_k=80, mbd=MbdMode.Enabled)
    d = cfg.to_dict()
    assert d["temperature"] == 1.2
    assert d["mbd"] == 1
    assert d["top_k"] == 80
    cfg_from_d = Config.from_dict(d)
    assert cfg_from_d.temperature == 1.2
    assert cfg_from_d.mbd == MbdMode.Enabled
    cfg_inst = Config()
    cfg_inst.from_dict(d)
    assert cfg_inst.temperature == 1.2
    assert cfg_inst.mbd == MbdMode.Enabled
    print("[PASS] Dict conversion verified.")


def test_json_methods():
    cfg = Config(temperature=0.75, mbd=MbdMode.Enabled, pause_duration=0.5)
    # dumpJson
    json_str = cfg.dumpJson()
    data = json.loads(json_str)
    assert data["temperature"] == 0.75
    assert data["mbd"] == 1
    assert data["pause_duration"] == 0.5
    # saveJson
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_name = tmp.name
    try:
        saved_path = cfg.saveJson(tmp_name)
        assert os.path.exists(saved_path)
        # loadJson as class factory
        cfg_loaded = Config.loadJson(saved_path)
        assert isinstance(cfg_loaded, Config)
        assert cfg_loaded.temperature == 0.75
        assert cfg_loaded.mbd == MbdMode.Enabled
        # loadJson as instance mutator
        cfg_inst = Config()
        cfg_inst.loadJson(saved_path)
        assert cfg_inst.temperature == 0.75
        assert cfg_inst.mbd == MbdMode.Enabled
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    # reset
    cfg_inst.temperature = 0.1
    cfg_inst.reset()
    assert cfg_inst.temperature == 1.0
    assert cfg_inst.mbd == MbdMode.Disabled
    print("[PASS] dumpJson, saveJson, loadJson, and reset verified.")


if __name__ == "__main__":
    test_enums()
    test_config_defaults()
    test_config_dicts()
    test_json_methods()
    print("\n🎉 ALL TESTS IN tests/test_config.py PASSED SUCCESSFULLY!")
