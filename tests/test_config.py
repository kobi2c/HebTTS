import base64
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
    SpeakerMode,
    Format,
    MbdMode,
    SeedMode,
    NikudMode,
    ChainMode,
    Speaker,
    LocalSpeaker,
    CloneSpeaker,
    DEFAULT_TEXT,
)


def test_enums():
    for enum_cls in (MbdMode, SeedMode, NikudMode, ChainMode):
        assert enum_cls.Disabled == 0
        assert enum_cls.Enabled == 1
        assert int(enum_cls.Disabled) == 0
        assert int(enum_cls.Enabled) == 1
        assert bool(enum_cls.Enabled) is True
        assert bool(enum_cls.Disabled) is False

    assert isinstance(Format.WAV, int)
    assert isinstance(Format.MP3, int)
    assert Format.WAV == 1
    assert Format.MP3 == 2
    assert int(Format.WAV) == 1
    assert int(Format.MP3) == 2
    assert Format.WAV == "wav"
    assert Format.MP3 == "mp3"
    assert Format("WAV") == Format.WAV
    assert Format("wav") == Format.WAV
    assert Format("MP3") == Format.MP3
    assert Format("mp3") == Format.MP3
    assert Format(1) == Format.WAV
    assert Format(2) == Format.MP3
    assert str(Format.WAV) == "wav"
    assert str(Format.MP3) == "mp3"
    assert Format.WAV.extension == ".wav"
    assert Format.MP3.extension == ".mp3"
    assert Format.extensions() == [".wav", ".mp3"]
    assert Format.names() == ["wav", "mp3"]
    assert Format.normalize(None) == [Format.WAV]
    assert Format.normalize(Format.MP3) == [Format.MP3]
    assert Format.normalize("mp3") == [Format.MP3]
    assert Format.normalize("wav") == [Format.WAV]
    assert Format.normalize("both") == [Format.WAV, Format.MP3]
    assert Format.normalize([Format.WAV, Format.MP3]) == [Format.WAV, Format.MP3]
    assert Format.normalize_formats("both") == [Format.WAV, Format.MP3]

    # SpeakerMode IntEnum
    assert isinstance(SpeakerMode.Local, int)
    assert isinstance(SpeakerMode.Clone, int)
    assert SpeakerMode.Local == 0
    assert SpeakerMode.Clone == 1
    assert int(SpeakerMode.Local) == 0
    assert int(SpeakerMode.Clone) == 1
    assert SpeakerMode("Local Speaker") == SpeakerMode.Local
    assert SpeakerMode("Custom Voice Clone") == SpeakerMode.Clone
    assert SpeakerMode("local") == SpeakerMode.Local
    assert SpeakerMode("clone") == SpeakerMode.Clone
    assert SpeakerMode(0) == SpeakerMode.Local
    assert SpeakerMode(1) == SpeakerMode.Clone
    assert str(SpeakerMode.Local) == "Local Speaker"
    assert str(SpeakerMode.Clone) == "Custom Voice Clone"
    assert SpeakerMode.Local == "Local Speaker"
    assert SpeakerMode.Clone == "Custom Voice Clone"
    try:
        SpeakerMode("predefined")
        assert False, "Should have raised ValueError for 'predefined'"
    except ValueError:
        pass
    print("[PASS] Enums verified.")


def test_config_defaults():
    cfg = Config()
    assert cfg.checkpoint == "checkpoint.pt"
    assert isinstance(cfg.speaker, Speaker)
    assert isinstance(cfg.speaker, LocalSpeaker)
    assert cfg.speaker.ui_mode == "Local Speaker"
    assert cfg.speaker.id in ("osim", "1")
    assert cfg.text == DEFAULT_TEXT
    assert cfg.pause_duration == 0.3
    assert cfg.temperature == 1.0
    assert cfg.top_k == 50
    assert cfg.mbd == MbdMode.Disabled
    assert cfg.seed == 420
    assert cfg.seed_mode == SeedMode.Enabled
    assert cfg.nikud_mode == NikudMode.Enabled
    assert cfg.chain_speaker == ChainMode.Disabled
    assert cfg.max_chunk_size == 100
    assert cfg.fade_ms == 20.0
    assert cfg.trim_noise is True
    assert cfg.pad_ms == 30.0
    # Zonos defaults
    assert cfg.model_choice == "notmax123/Zonos-Hebrew"
    assert cfg.language == "he"
    assert cfg.fmax == 24000.0
    assert cfg.pitch_std == 45.0
    assert cfg.speaking_rate == 15.0
    assert cfg.dnsmos_ovrl == 4.0
    assert cfg.speaker_noised is False
    assert cfg.cfg_scale == 2.0
    assert cfg.vq_score == 0.78
    assert cfg.emotions == [0.0] * 8
    print("[PASS] Config defaults verified.")


def test_config_dicts():
    cfg = Config(
        temperature=1.2,
        top_k=80,
        mbd=MbdMode.Enabled,
        chain_speaker=ChainMode.Enabled,
        max_chunk_size=75,
        fade_ms=15.0,
        trim_noise=False,
        pad_ms=45.0,
        model_choice="Zyphra/Zonos-v0.1-transformer",
        language="en-us",
        fmax=22050.0,
        pitch_std=50.0,
        speaking_rate=18.0,
        dnsmos_ovrl=4.5,
        speaker_noised=True,
        cfg_scale=3.0,
        vq_score=0.82,
        emotions=[0.1] * 8,
    )
    d = cfg.to_dict()
    assert d["temperature"] == 1.2
    assert d["mbd"] == 1
    assert d["top_k"] == 80
    assert d["chain_speaker"] == 1
    assert d["max_chunk_size"] == 75
    assert d["fade_ms"] == 15.0
    assert d["trim_noise"] is False
    assert d["pad_ms"] == 45.0
    assert d["seed"] == 420
    assert d["seed_mode"] == 1
    assert d["nikud_mode"] == 1
    assert d["model_choice"] == "Zyphra/Zonos-v0.1-transformer"
    assert d["language"] == "en-us"
    assert d["fmax"] == 22050.0
    assert d["pitch_std"] == 50.0
    assert d["speaking_rate"] == 18.0
    assert d["dnsmos_ovrl"] == 4.5
    assert d["speaker_noised"] is True
    assert d["cfg_scale"] == 3.0
    assert d["vq_score"] == 0.82
    assert d["emotions"] == [0.1] * 8
    assert "speaker_mode" not in d
    assert "clone" not in d
    assert "speaker" in d
    assert isinstance(d["speaker"], dict)
    assert d["speaker"]["mode"] == 0
    assert d["speaker"]["id"] in ("osim", "1")

    cfg_from_d = Config.from_dict(d)
    assert cfg_from_d.temperature == 1.2
    assert cfg_from_d.mbd == MbdMode.Enabled
    assert cfg_from_d.chain_speaker == ChainMode.Enabled
    assert cfg_from_d.max_chunk_size == 75
    assert cfg_from_d.fade_ms == 15.0
    assert cfg_from_d.trim_noise is False
    assert cfg_from_d.pad_ms == 45.0
    assert cfg_from_d.seed == 420
    assert cfg_from_d.seed_mode == SeedMode.Enabled
    assert cfg_from_d.nikud_mode == NikudMode.Enabled
    assert cfg_from_d.model_choice == "Zyphra/Zonos-v0.1-transformer"
    assert cfg_from_d.language == "en-us"
    assert cfg_from_d.fmax == 22050.0
    assert cfg_from_d.pitch_std == 50.0
    assert cfg_from_d.speaking_rate == 18.0
    assert cfg_from_d.dnsmos_ovrl == 4.5
    assert cfg_from_d.speaker_noised is True
    assert cfg_from_d.cfg_scale == 3.0
    assert cfg_from_d.vq_score == 0.82
    assert cfg_from_d.emotions == [0.1] * 8
    assert isinstance(cfg_from_d.speaker, Speaker)
    assert isinstance(cfg_from_d.speaker, LocalSpeaker)

    # Test with custom clone speaker
    clone_spk = CloneSpeaker(id="custom", filename="custom.wav", text="שלום")
    cfg_clone = Config(speaker=clone_spk)
    d_clone = cfg_clone.to_dict()
    assert d_clone["speaker"]["id"] == "custom"
    assert d_clone["speaker"]["mode"] == 1
    assert d_clone["speaker"]["filename"] == "custom.wav"
    assert d_clone["speaker"]["text"] == "שלום"
    cfg_restored = Config.from_dict(d_clone)
    assert isinstance(cfg_restored.speaker, Speaker)
    assert isinstance(cfg_restored.speaker, CloneSpeaker)
    assert cfg_restored.speaker.id == "custom"

    cfg_inst = Config()
    cfg_inst.from_dict(d)
    assert cfg_inst.temperature == 1.2
    assert cfg_inst.mbd == MbdMode.Enabled
    assert cfg_inst.chain_speaker == ChainMode.Enabled
    assert cfg_inst.max_chunk_size == 75
    assert cfg_inst.seed == 420
    assert cfg_inst.seed_mode == SeedMode.Enabled
    assert cfg_inst.nikud_mode == NikudMode.Enabled
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
    cfg_inst.max_chunk_size = 50
    cfg_inst.reset()
    assert cfg_inst.temperature == 1.0
    assert cfg_inst.mbd == MbdMode.Disabled
    assert cfg_inst.max_chunk_size == 100
    print("[PASS] dumpJson, saveJson, loadJson, and reset verified.")


def test_standalone_speaker_config():
    fake_audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    b64_str = base64.b64encode(fake_audio_bytes).decode("utf-8")
    cfg_dict = {
        "speaker": {
            "mode": 1,
            "id": "remote_clone",
            "filename": "non_existent_file.wav",
            "ext": "wav",
            "text": "טקסט קול מרוחק",
            "data": b64_str,
        }
    }
    cfg = Config.from_dict(cfg_dict)
    assert isinstance(cfg.speaker, Speaker)
    assert isinstance(cfg.speaker, CloneSpeaker)
    assert cfg.speaker.ui_mode == "Custom Voice Clone"
    assert cfg.speaker.id == "remote_clone"
    assert not os.path.exists(cfg.speaker.filename)
    # resolve_audio_path extracts base64 data to a valid temp file
    temp_path = cfg.speaker.resolve_audio_path()
    assert os.path.exists(temp_path)
    with open(temp_path, "rb") as f:
        assert f.read() == fake_audio_bytes
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print("[PASS] Standalone speaker config serialization and extraction verified.")


if __name__ == "__main__":
    test_enums()
    test_config_defaults()
    test_config_dicts()
    test_json_methods()
    test_standalone_speaker_config()
    print("\n🎉 ALL TESTS IN tests/test_config.py PASSED SUCCESSFULLY!")
