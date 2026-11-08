import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import Speaker, SpeakerData, SpeakerStore


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
    # Test Speaker.audio_file property
    assert s1.audio_file == s1.wav
    assert sp_test.audio_file == "custom_spk.wav"

    # Test auto-discovery: .wav without .txt must NOT be discovered
    dummy_wav = Path("speakers/dummy_temp_spk.wav")
    dummy_txt = Path("speakers/dummy_temp_spk.txt")
    dummy_mp3 = Path("speakers/dummy_temp_mp3_spk.mp3")
    dummy_mp3_txt = Path("speakers/dummy_temp_mp3_spk.txt")
    collision_wav = Path("speakers/dummy_collision.wav")
    collision_mp3 = Path("speakers/dummy_collision.mp3")
    collision_txt = Path("speakers/dummy_collision.txt")
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

        # Test MP3 auto-discovery
        dummy_mp3.touch()
        dummy_mp3_txt.write_text("Hello MP3 text", encoding="utf-8")
        mgr.clear()
        discovered_mp3 = mgr.find("dummy_temp_mp3_spk")
        assert discovered_mp3 is not None
        assert discovered_mp3.id == "dummy_temp_mp3_spk"
        assert discovered_mp3.wav == "dummy_temp_mp3_spk.mp3"
        assert discovered_mp3.audio_file == "dummy_temp_mp3_spk.mp3"
        assert discovered_mp3.txt == "Hello MP3 text"

        # Test collision: if both .wav and .mp3 exist, .wav takes precedence
        collision_wav.touch()
        collision_mp3.touch()
        collision_txt.write_text("Collision prompt", encoding="utf-8")
        mgr.clear()
        discovered_coll = mgr.find("dummy_collision")
        assert discovered_coll is not None
        assert discovered_coll.wav == "dummy_collision.wav"
    finally:
        for p in (dummy_wav, dummy_txt, dummy_mp3, dummy_mp3_txt, collision_wav, collision_mp3, collision_txt):
            if p.exists():
                p.unlink()
        mgr.clear()
    mgr.clear()
    assert mgr._list is None
    print("[PASS] SpeakerData (load, list, find, clear) verified.")


if __name__ == "__main__":
    test_speaker_data()
    print("\n🎉 ALL TESTS IN tests/test_speaker.py PASSED SUCCESSFULLY!")
