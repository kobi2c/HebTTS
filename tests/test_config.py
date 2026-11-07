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
    CheckpointStore,
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
    # Test split_text with default max_len=100
    from make_voice import split_text
    long_hebrew = (
        "היום אנחנו הולכים ללמוד על העולם הקסום של כללי ההתחלקות, "
        "אותם קיצורי דרך נחמדים שמצילים אותנו מהמטלה המייגעת לחלוטין של חילוק ארוך. "
        "האם יצא לכם להביט במספר ענק ולדעת מיד אם אפשר לפצל אותו בשלמות לקבוצות קטנות יותר?"
    )
    chunks = split_text(long_hebrew)
    for c in chunks:
        assert len(c) <= 100, f"Chunk exceeded max_len=100: {len(c)} chars ({c})"
    assert len(chunks) >= 2
    # Test NoiseFix.fade and NoiseFix.Fade
    from config import NoiseFix
    import numpy as np
    raw_audio = np.ones(2400, dtype=np.float32)  # 100ms at 24kHz
    faded = NoiseFix.fade(raw_audio, sr=24000, fade_in_ms=20.0, fade_out_ms=20.0)
    assert faded[0] == 0.0, f"Expected 0.0 at start, got {faded[0]}"
    assert faded[-1] == 0.0, f"Expected 0.0 at end, got {faded[-1]}"
    assert np.isclose(faded[480], 1.0, atol=1e-3), "Expected mid-body to be 1.0"
    assert np.array_equal(NoiseFix.fade(raw_audio, fade_in_ms=0, fade_out_ms=0), raw_audio)
    assert len(NoiseFix.fade(np.array([], dtype=np.float32))) == 0

    # Test NoiseFix.fade
    assert len(NoiseFix.trim(np.array([], dtype=np.float32))) == 0
    zeros = np.zeros(2400, dtype=np.float32)
    assert np.array_equal(NoiseFix.trim(zeros), zeros)

    # Synthetic signal: 200ms noise (peak 0.02) + 50ms silence + 500ms speech tone (peak 0.5)
    sr = 24000
    noise = np.random.uniform(-0.02, 0.02, int(sr * 0.20)).astype(np.float32)
    silence = np.zeros(int(sr * 0.05), dtype=np.float32)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    test_sig = np.concatenate([noise, silence, speech])
    trimmed = NoiseFix.trim(test_sig, sr=sr, pad_ms=30.0)
    cut_ms = ((len(test_sig) - len(trimmed)) / sr) * 1000.0
    assert 200.0 <= cut_ms <= 230.0, f"Expected cut between 200ms and 230ms, got {cut_ms}ms"

    print("[PASS] VoiceMaker.parse_mki, split_text, apply_fade, and trim_leading_noise verified.")

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
    assert mgr._main_path == "speakers/speakers.yaml"
    speakers = mgr.list()
    assert isinstance(speakers, list) and len(speakers) >= 3
    assert all(isinstance(s, Speaker) for s in speakers)
    # First speaker: osim
    s1 = speakers[0]
    assert s1.id == "osim"
    assert s1.wav == "osim.wav"
    assert s1.base == "speakers"
    assert s1.full_path == Path("speakers/osim.wav")
    assert s1.exists is True
    assert len(s1.txt) > 0
    # find by 1-based int, string digit, and id
    found_by_int = mgr.find(1)
    found_by_str_digit = mgr.find("1")
    found_by_id = mgr.find("osim")
    found_by_wav = mgr.find("osim.wav")
    assert found_by_int is s1
    assert found_by_str_digit is s1
    assert found_by_id is s1
    assert found_by_wav is s1
    # find second speaker: geek
    s2 = mgr.find(2)
    assert s2 is not None and s2.id == "geek"
    assert s2.wav == "geek.wav"
    assert s2.base == "speakers"
    assert s2.full_path == Path("speakers/geek.wav")
    # Negative lookup cases
    assert mgr.find(999) is None
    assert mgr.find("0") is None
    assert mgr.find(-1) is None
    assert mgr.find("unknown_speaker_xyz") is None
    # Test Speaker.parse
    sp_test = Speaker(id="custom", wav="")
    sp_test.parse(
        {"audio-prompt": "custom_spk.wav", "text-prompt": "Custom speaker text"},
        Path("speakers"),
        "custom",
    )
    assert sp_test.id == "custom"
    assert sp_test.wav == "custom_spk.wav"
    assert sp_test.base == "speakers"
    assert sp_test.txt == "Custom speaker text"
    assert sp_test.full_path == Path("speakers/custom_spk.wav")
    # Test Speaker.parse with absolute path inside speakers directory
    abs_wav_test = (Path("speakers").resolve() / "abs_spk.wav").as_posix()
    sp_abs = Speaker(id="abs_spk")
    sp_abs.parse(
        {"audio-prompt": abs_wav_test, "text-prompt": "Absolute text"},
        Path("speakers"),
        "abs_spk",
    )
    assert sp_abs.wav == "abs_spk.wav"
    assert sp_abs.full_path == Path("speakers/abs_spk.wav")
    # Test auto-discovery: .wav without .txt must NOT be discovered
    dummy_wav = Path("speakers/dummy_temp_spk.wav")
    dummy_txt = Path("speakers/dummy_temp_spk.txt")
    try:
        dummy_wav.touch()
        mgr.clear()
        assert mgr.find("dummy_temp_spk") is None
        # When companion .txt exists with text, it MUST be discovered
        dummy_txt.write_text("Hello prompt text", encoding="utf-8")
        mgr.clear()
        discovered = mgr.find("dummy_temp_spk")
        assert discovered is not None
        assert discovered.id == "dummy_temp_spk"
        assert discovered.wav == "dummy_temp_spk.wav"
        assert discovered.txt == "Hello prompt text"
        assert discovered.full_path == Path("speakers/dummy_temp_spk.wav")
    finally:
        if dummy_wav.exists():
            dummy_wav.unlink()
        if dummy_txt.exists():
            dummy_txt.unlink()
        mgr.clear()
    mgr.clear()
    assert mgr._list is None
    print("[PASS] SpeakerData (load, list, find, clear) verified.")

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

def test_table_printing():
    import io
    from contextlib import redirect_stdout
    from sample import Sampler
    from config import CheckpointStore, SpeakerStore
    s = Sampler.__new__(Sampler)
    s.checkpoints = CheckpointStore()
    s.speakers = SpeakerStore()
    f = io.StringIO()
    with redirect_stdout(f):
        s.print_checkpoints()
    output = f.getvalue()
    assert "Available checkpoints:" in output
    assert "Checkpoint ID" in output
    assert "Path" in output
    assert "Status" in output
    assert "Description" in output
    f = io.StringIO()
    with redirect_stdout(f):
        s.print_speakers()
    output = f.getvalue()
    assert "Available speakers:" in output
    assert "ID" in output
    assert "Audio-File" in output
    assert "Audio-Text" in output
    print("[PASS] Table printing (checkpoints and speakers) verified.")

def test_range_and_range_list():
    from config import Range, RangeList
    # 1. Range parsing
    r1 = Range.parse("5")
    assert r1.start == 5 and r1.end == 5
    assert str(r1) == "5"
    r2 = Range.parse("2-6")
    assert r2.start == 2 and r2.end == 6
    assert str(r2) == "2-6"
    # Inverted bounds auto-swap
    r_inv = Range.parse("6-2")
    assert r_inv.start == 2 and r_inv.end == 6
    # Wildcards: @-Y, x-Y, X-Y
    r_at_start = Range.parse("@-5")
    assert r_at_start.start is None and r_at_start.end == 5
    assert str(r_at_start) == "@-5"
    # Wildcards: Y-@, Y-x, Y-X
    r_at_end = Range.parse("10-@")
    assert r_at_end.start == 10 and r_at_end.end is None
    assert str(r_at_end) == "10-@"
    # Invalid formats rejected
    for invalid in ["-5", "5-", "abc", "0", "-1"]:
        try:
            Range.parse(invalid)
            assert False, f"Expected ValueError for '{invalid}'"
        except ValueError:
            pass
    # 2. Resolution and containment
    assert r_at_start.resolve(10) == (1, 5)
    assert r_at_end.resolve(10) == (10, 10)
    assert r2.resolve(10) == (2, 6)
    assert Range(15, 20).resolve(10) == (10, 10)
    assert Range().resolve(10) == (1, 10)
    assert Range().resolve(0) == (1, 1)
    assert Range(0, 15).resolve(10) == (1, 10)
    assert Range(-5, 0).resolve(10) == (1, 1)
    assert r2.resolve(0) == (1, 1)
    assert r2.contains(2) is True
    assert r2.contains(4) is True
    assert r2.contains(6) is True
    assert r2.contains(1) is False
    assert r2.contains(7) is False
    assert 4 in r2
    assert 1 not in r2
    # 3. RangeList parsing and normalization
    rl = RangeList.parse("10-12, 1-3, 5, 2-4, @-2")
    norm = rl.normalize(last=15)
    # Merges @-2, 1-3, 2-4, 5 -> [1, 5] and 10-12 -> [10, 12]
    assert len(norm.ranges) == 2
    assert norm.ranges[0] == Range(1, 5)
    assert norm.ranges[1] == Range(10, 12)
    # Contiguous range merging: [1, 3] and [4, 6] -> [1, 6]
    rl_cont = RangeList.parse("1-3, 4-6").normalize(last=10)
    assert len(rl_cont.ranges) == 1
    assert rl_cont.ranges[0] == Range(1, 6)
    # Wildcard lists with resolution
    rl_wild = RangeList.parse("@-3, 8-@, 5").normalize(last=10)
    assert len(rl_wild.ranges) == 3
    assert rl_wild.ranges[0] == Range(1, 3)
    assert rl_wild.ranges[1] == Range(5, 5)
    assert rl_wild.ranges[2] == Range(8, 10)
    # indices()
    assert rl_wild.indices(10) == [1, 2, 3, 5, 8, 9, 10]
    # contains()
    assert rl_wild.contains(1) is True
    assert rl_wild.contains(4) is False
    assert rl_wild.contains(5) is True
    assert rl_wild.contains(8) is True
    assert 3 in rl_wild
    assert 4 not in rl_wild
    # 4. make_voice parser integration
    from make_voice import VoiceMaker
    vm = VoiceMaker()
    parser = vm.create_parser()
    args = parser.parse_args(["-i", "test.mki", "--range", "1-5, 8-@"])
    assert args.range == "1-5, 8-@"
    print("[PASS] Range and RangeList verified.")

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
    test_table_printing()
    test_range_and_range_list()
    print("\n🎉 ALL TESTS IN tests/test_config.py PASSED SUCCESSFULLY!")
