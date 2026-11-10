import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import Speaker, LocalSpeaker, CloneSpeaker, SpeakerStore, SpeakerMode


def test_speaker_data():
    mgr = SpeakerStore()
    assert mgr._list is None
    assert mgr._main_path == "speakers/speakers.yaml"
    speakers = mgr.list()
    assert isinstance(speakers, list) and len(speakers) >= 3
    assert all(isinstance(s, LocalSpeaker) for s in speakers)
    assert all(isinstance(s, Speaker) for s in speakers)

    # First speaker: osim
    s1 = speakers[0]
    assert s1.id == "osim"
    assert s1.filename == "osim.wav"
    assert s1.base == "speakers"
    assert s1.full_path == Path("speakers/osim.wav")
    assert s1.exists is True
    assert len(s1.text) > 0
    assert s1.ui_mode == "Local Speaker"
    assert s1.to_dict()["mode"] == 0

    # Polymorphic prompt resolution & validation for LocalSpeaker
    s1.validate()
    prompt_path, prompt_text = s1.resolve_prompt()
    assert prompt_path.endswith("osim.wav")
    assert prompt_text == s1.text
    assert s1.ui_preview() == (s1.id, None, "")

    # Test CloneSpeaker serialization and polymorphism
    clone_s = CloneSpeaker(id="cl1", filename="osim.wav", text="clone text")
    assert clone_s.ui_mode == "Custom Voice Clone"
    clone_d = clone_s.to_dict(embed_data=False)
    assert clone_d["mode"] == 1
    clone_restored = Speaker.from_dict(clone_d)
    assert isinstance(clone_restored, CloneSpeaker)
    assert clone_restored.ui_mode == "Custom Voice Clone"
    assert clone_restored == clone_s

    # Equality requires same class
    pred_s = LocalSpeaker(id="cl1", filename="osim.wav", text="clone text")
    assert pred_s != clone_s

    # CloneSpeaker validation and prompt resolution
    clone_audio_path, clone_audio_text = clone_s.resolve_prompt()
    assert clone_audio_path.endswith("osim.wav")
    assert clone_audio_text == "clone text"

    # Validation errors on invalid CloneSpeaker
    invalid_clone = CloneSpeaker(id="bad", text="")
    try:
        invalid_clone.resolve_prompt()
        assert False, "Should have raised ValueError on missing audio/text"
    except ValueError:
        pass

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
    assert s2.filename == "geek.wav"
    assert s2.base == "speakers"
    assert s2.full_path == Path("speakers/geek.wav")

    # Negative lookup cases
    assert mgr.find(999) is None
    assert mgr.find("0") is None
    assert mgr.find(-1) is None
    assert mgr.find("unknown_speaker_xyz") is None

    # Test LocalSpeaker.parse
    sp_test = LocalSpeaker(id="custom", filename="")
    sp_test.parse(
        {"audio-prompt": "custom_spk.wav", "text-prompt": "Custom speaker text"},
        Path("speakers"),
        "custom",
    )
    assert sp_test.id == "custom"
    assert sp_test.filename == "custom_spk.wav"
    assert sp_test.base == "speakers"
    assert sp_test.text == "Custom speaker text"
    assert sp_test.full_path == Path("speakers/custom_spk.wav")

    # Test Speaker.parse with absolute path inside speakers directory
    abs_wav_test = (Path("speakers").resolve() / "abs_spk.wav").as_posix()
    sp_abs = LocalSpeaker(id="abs_spk")
    sp_abs.parse(
        {"audio-prompt": abs_wav_test, "text-prompt": "Absolute text"},
        Path("speakers"),
        "abs_spk",
    )
    assert sp_abs.filename == "abs_spk.wav"
    assert sp_abs.full_path == Path("speakers/abs_spk.wav")

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
        assert isinstance(discovered, LocalSpeaker)
        assert discovered.id == "dummy_temp_spk"
        assert discovered.filename == "dummy_temp_spk.wav"
        assert discovered.text == "Hello prompt text"
        assert discovered.full_path == Path("speakers/dummy_temp_spk.wav")

        # Test MP3 auto-discovery
        dummy_mp3.touch()
        dummy_mp3_txt.write_text("Hello MP3 text", encoding="utf-8")
        mgr.clear()
        discovered_mp3 = mgr.find("dummy_temp_mp3_spk")
        assert discovered_mp3 is not None
        assert isinstance(discovered_mp3, LocalSpeaker)
        assert discovered_mp3.id == "dummy_temp_mp3_spk"
        assert discovered_mp3.filename == "dummy_temp_mp3_spk.mp3"
        assert discovered_mp3.text == "Hello MP3 text"

        # Test collision: if both .wav and .mp3 exist, .wav takes precedence
        collision_wav.touch()
        collision_mp3.touch()
        collision_txt.write_text("Collision prompt", encoding="utf-8")
        mgr.clear()
        discovered_coll = mgr.find("dummy_collision")
        assert discovered_coll is not None
        assert isinstance(discovered_coll, LocalSpeaker)
        assert discovered_coll.filename == "dummy_collision.wav"
    finally:
        for p in (dummy_wav, dummy_txt, dummy_mp3, dummy_mp3_txt, collision_wav, collision_mp3, collision_txt):
            if p.exists():
                p.unlink()
        mgr.clear()
    mgr.clear()
    assert mgr._list is None

    # Test SpeakerStore.default() and SpeakerStore.resolve()
    def_spk = SpeakerStore.default()
    assert isinstance(def_spk, LocalSpeaker)
    assert def_spk.id == "osim"
    assert mgr.default().id == "osim"

    # SpeakerStore.resolve
    assert SpeakerStore.resolve(None).id == "osim"
    assert SpeakerStore.resolve(1).id == "osim"
    assert SpeakerStore.resolve("osim").id == "osim"
    assert SpeakerStore.resolve(def_spk) is def_spk
    resolved_dict = SpeakerStore.resolve({"id": "custom_dict", "filename": "c.wav", "text": "hello", "mode": 0})
    assert isinstance(resolved_dict, LocalSpeaker)
    assert resolved_dict.id == "custom_dict"
    assert resolved_dict.filename == "c.wav"
    assert resolved_dict.text == "hello"
    resolved_missing = SpeakerStore.resolve("non_existent_voice")
    assert isinstance(resolved_missing, LocalSpeaker)
    assert resolved_missing.id == "non_existent_voice"
    assert SpeakerStore.find("geek").id == "geek"

    print("[PASS] SpeakerStore (load, list, find, default, resolve, clear) verified.")


def test_speaker_create_factory():
    # 1. Local resolution via Speaker.create
    spk_local = Speaker.create(mode=SpeakerMode.Local, id="osim")
    assert isinstance(spk_local, LocalSpeaker)
    assert spk_local.id == "osim"

    # Dedicated factory LocalSpeaker.create
    spk_local_direct = LocalSpeaker.create(id="osim")
    assert isinstance(spk_local_direct, LocalSpeaker)
    assert spk_local_direct.id == "osim"

    # Local ignores clone_text and only uses id to fetch from store
    spk_local_override = Speaker.create(mode=SpeakerMode.Local, id="osim", clone_text="ignored text")
    assert spk_local_override.id == "osim"
    assert spk_local_override.text == "אני משתדל מאוד שכל חסות תהיה מיוחדת ושונה"

    # Local with None id -> default speaker
    spk_local_def = Speaker.create(mode=SpeakerMode.Local)
    assert isinstance(spk_local_def, LocalSpeaker)
    assert spk_local_def.id == "osim"

    # 2. Clone with empty / None clone_path
    spk_clone_empty = Speaker.create(mode=SpeakerMode.Clone, clone_text="empty clone text")
    assert isinstance(spk_clone_empty, CloneSpeaker)
    assert spk_clone_empty.id == "clone"
    assert spk_clone_empty.text == "empty clone text"

    # Dedicated factory CloneSpeaker.create
    spk_clone_direct = CloneSpeaker.create(clone_text="direct clone text")
    assert isinstance(spk_clone_direct, CloneSpeaker)
    assert spk_clone_direct.text == "direct clone text"

    # Clone with custom id and no path
    spk_clone_custom_id = Speaker.create(mode=SpeakerMode.Clone, id="custom_id", clone_text="custom id text")
    assert isinstance(spk_clone_custom_id, CloneSpeaker)
    assert spk_clone_custom_id.id == "custom_id"
    assert spk_clone_custom_id.text == "custom id text"

    # 3. Clone with file path
    dummy_audio = Path("speakers/osim.wav")  # existing audio
    spk_clone = Speaker.create(mode=SpeakerMode.Clone, clone_path=dummy_audio, clone_text="clone audio text", load_audio=True)
    assert isinstance(spk_clone, CloneSpeaker)
    assert spk_clone.id == "osim"
    assert spk_clone.filename == "osim.wav"
    assert spk_clone.text == "clone audio text"
    assert len(spk_clone.data) > 0  # base64 loaded

    # 4. Clone from dict source
    spk_clone_dict = Speaker.create(
        mode=SpeakerMode.Clone,
        source={"id": "from_dict", "filename": "fd.wav", "text": "dict text"},
    )
    assert isinstance(spk_clone_dict, CloneSpeaker)
    assert spk_clone_dict.id == "from_dict"
    assert spk_clone_dict.filename == "fd.wav"

    # 5. Clone from existing LocalSpeaker
    existing_spk = LocalSpeaker(id="orig", filename="orig.wav")
    converted_spk = Speaker.create(mode=SpeakerMode.Clone, speaker=existing_spk, clone_text="new text")
    assert isinstance(converted_spk, CloneSpeaker)
    assert converted_spk.id == "orig"
    assert converted_spk.text == "new text"

    # 6. Verify SpeakerMode backward compatibility is removed
    assert not hasattr(SpeakerMode, "Predefined")
    try:
        SpeakerMode("predefined")
        assert False, "Should have raised ValueError on 'predefined'"
    except ValueError:
        pass

    print("[PASS] Speaker.create factory verified.")


if __name__ == "__main__":
    test_speaker_data()
    test_speaker_create_factory()
    print("\n🎉 ALL TESTS IN tests/test_speaker.py PASSED SUCCESSFULLY!")
