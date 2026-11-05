import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Gracefully mock container-only dependencies if running in a lightweight host environment
for mod in [
    "omegaconf",
    "soundfile",
    "gradio",
    "torch",
    "torchaudio",
    "valle",
    "valle.data",
    "valle.data.collation",
    "valle.data.hebrew_root_tokenizer",
    "valle.models",
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from config import (
    Config,
    MbdMode,
    RandomMode,
    NikudMode,
    DEFAULT_TEXT,
    audio_to_base64,
    base64_to_audio,
    Speaker,
    SpeakerData,
    Checkpoint,
    CheckpointData,
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


def test_mki_parsing_with_config():
    from make_voice import VoiceMaker

    vm = VoiceMaker()
    base_cfg = Config(predefined_speaker="osim", temperature=1.0)
    lines = [
        "[This is a comment]",
        "@{\"predefined_speaker\": \"speaker2\", \"temperature\": 0.6}",
        "שלום שורה ראשונה",
        "!{\"predefined_speaker\": \"speaker3\"}",
        "שלום שורה שנייה",
    ]
    blocks = vm.parse_mki(lines, base_cfg)
    assert len(blocks) == 2

    # Block 1: incremental update
    b1_cfg, b1_text = blocks[0]
    assert b1_cfg.predefined_speaker == "speaker2"
    assert b1_cfg.temperature == 0.6
    assert b1_text == "שלום שורה ראשונה"

    # Block 2: reset to base + update
    b2_cfg, b2_text = blocks[1]
    assert b2_cfg.predefined_speaker == "speaker3"
    assert b2_cfg.temperature == 1.0  # reset back to base_cfg
    assert b2_text == "שלום שורה שנייה"
    print("[PASS] VoiceMaker.parse_mki verified.")


def test_generate_audio_strict_config():
    from GradioPage import GradioPage

    gp = GradioPage()
    # Test that passing a non-Config raises TypeError
    try:
        gp.generate_audio({"checkpoint": "test.pt"})
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "accepts only a Config instance" in str(e)
    print("[PASS] GradioPage.generate_audio strict Config enforcement verified.")


def test_gradio_page_config_helpers():
    from GradioPage import GradioPage

    gp = GradioPage()
    assert isinstance(gp.config, Config)
    # Check default ui values
    ui_vals = gp.get_ui_values_from_config()
    assert len(ui_vals) == 13
    assert ui_vals[0] == gp.config.checkpoint
    assert ui_vals[1] == "Predefined Speaker"
    assert ui_vals[5] == DEFAULT_TEXT
    assert ui_vals[6] == 0.3
    assert ui_vals[7] == 1.0
    assert ui_vals[8] == 50
    assert ui_vals[9] == 0  # int(MbdMode.Disabled)
    assert ui_vals[10] == 420
    assert ui_vals[11] == 1  # int(RandomMode.Enabled)
    assert ui_vals[12] == 1  # int(NikudMode.Enabled)

    # Test update_config_from_ui
    gp.update_config_from_ui(
        checkpoint="custom_cp.pt",
        speaker_mode="Custom Voice Clone",
        predefined_speaker="geek",
        clone_audio="ref.wav",
        clone_text="ref text",
        text="custom text",
        pause_duration=0.5,
        temperature=0.8,
        top_k=40,
        mbd=1,
        seed_number=123,
        randomize_seed=0,
        nikud_converter=0,
    )
    assert gp.config.checkpoint == "custom_cp.pt"
    assert gp.config.speaker_mode == "Custom Voice Clone"
    assert gp.config.predefined_speaker == "geek"
    assert gp.config.clone_audio == "ref.wav"
    assert gp.config.clone_text == "ref text"
    assert gp.config.text == "custom text"
    assert gp.config.pause_duration == 0.5
    assert gp.config.temperature == 0.8
    assert gp.config.top_k == 40
    assert gp.config.mbd == MbdMode.Enabled
    assert gp.config.seed_number == 123
    assert gp.config.randomize_seed == RandomMode.Disabled
    assert gp.config.nikud_converter == NikudMode.Disabled

    # Export & Import settings
    exported_json = gp.export_settings()
    assert os.path.exists(exported_json)
    try:
        # Reset and import
        gp.config.reset()
        assert gp.config.temperature == 1.0
        imported_tuple = gp.import_settings(exported_json)
        assert len(imported_tuple) == 13
        assert gp.config.temperature == 0.8
        assert gp.config.checkpoint == "custom_cp.pt"
    finally:
        if os.path.exists(exported_json):
            os.remove(exported_json)

    print("[PASS] GradioPage config integration verified.")


def test_speaker_data():
    mgr = SpeakerData()
    assert mgr._list is None
    assert mgr.data_path == "speakers/speakers.yaml"

    speakers = mgr.list()
    assert isinstance(speakers, list) and len(speakers) >= 3
    assert all(isinstance(s, Speaker) for s in speakers)

    # First speaker: osim
    s1 = speakers[0]
    assert s1.id == "osim"
    assert s1.wav_base == "osim.wav"
    assert "speakers" in s1.wav and s1.wav.endswith("osim.wav")
    assert len(s1.txt) > 0

    # find by 1-based int, string digit, and id
    found_by_int = mgr.find(1)
    found_by_str_digit = mgr.find("1")
    found_by_id = mgr.find("osim")
    found_by_id_upper = mgr.find("OSIM")
    assert found_by_int is s1
    assert found_by_str_digit is s1
    assert found_by_id is s1
    assert found_by_id_upper is s1

    # find second speaker: geek
    s2 = mgr.find(2)
    assert s2 is not None and s2.id == "geek"
    assert s2.wav_base == "geek.wav"

    # Negative lookup cases
    assert mgr.find(999) is None
    assert mgr.find("0") is None
    assert mgr.find(-1) is None
    assert mgr.find("unknown_speaker_xyz") is None

    mgr.clear()
    assert mgr._list is None

    print("[PASS] SpeakerData (load, list, find, clear) verified.")


def test_checkpoint_data():
    mgr = CheckpointData()
    assert mgr._list is None
    assert mgr.data_path == "checkpoints/checkpoints.yaml"

    checkpoints = mgr.list()
    assert isinstance(checkpoints, list) and len(checkpoints) >= 1
    assert all(isinstance(c, Checkpoint) for c in checkpoints)

    # First checkpoint: checkpoint.pt
    c1 = checkpoints[0]
    assert c1.id == "checkpoint.pt"
    assert c1.path == "checkpoint.pt"
    assert c1.full_path == Path("checkpoints/checkpoint.pt")
    assert c1.gdrive_id == "11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv"
    assert len(c1.info) > 0
    assert isinstance(c1.exists, bool)

    # find by 1-based int, string digit, and id/path
    found_by_int = mgr.find(1)
    found_by_str_digit = mgr.find("1")
    found_by_id = mgr.find("checkpoint.pt")
    found_by_id_upper = mgr.find("CHECKPOINT.PT")
    found_by_full_path = mgr.find("checkpoints/checkpoint.pt")
    assert found_by_int is c1
    assert found_by_str_digit is c1
    assert found_by_id is c1
    assert found_by_id_upper is c1
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

    mgr.clear()
    assert mgr._list is None

    print("[PASS] CheckpointData (load, list, find, clear) verified.")


if __name__ == "__main__":
    test_enums()
    test_config_defaults()
    test_config_dicts()
    test_json_methods()
    test_mki_parsing_with_config()
    test_generate_audio_strict_config()
    test_gradio_page_config_helpers()
    test_speaker_data()
    test_checkpoint_data()
    print("\n🎉 ALL TESTS IN tests/test_config.py PASSED SUCCESSFULLY!")
