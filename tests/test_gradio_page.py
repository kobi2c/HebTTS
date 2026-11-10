import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import (
    Config,
    SpeakerMode,
    Speaker,
    LocalSpeaker,
    CloneSpeaker,
    MbdMode,
    RandomMode,
    NikudMode,
    DEFAULT_TEXT,
    CheckpointStore,
    SpeakerStore,
)
from GradioPage import GradioPage
from make_sample import SampleMaker


def test_generate_audio_strict_config():
    gp = GradioPage()
    # Test that passing a non-Config raises TypeError
    try:
        gp.generate_audio({"checkpoint": "test.pt"})
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "accepts only a Config instance" in str(e)
    print("[PASS] GradioPage.generate_audio strict Config enforcement verified.")


def test_gradio_page_config_helpers():
    gp = GradioPage()
    assert isinstance(gp.config, Config)
    # Check default ui values
    ui_vals = gp.get_ui_values_from_config()
    assert len(ui_vals) == 14
    assert ui_vals[0] == gp.config.checkpoint
    assert ui_vals[1] == "Local Speaker"
    assert ui_vals[2] in ("osim", "1")
    assert ui_vals[3] is None
    assert ui_vals[4] == ""
    assert ui_vals[5] == DEFAULT_TEXT
    assert ui_vals[6] == 0.3
    assert ui_vals[7] == 1.0
    assert ui_vals[8] == 50
    assert ui_vals[9] == 0  # int(MbdMode.Disabled)
    assert ui_vals[10] == 420
    assert ui_vals[11] == 1  # int(RandomMode.Enabled)
    assert ui_vals[12] == 1  # int(NikudMode.Enabled)
    assert ui_vals[13] == 100
    # Test update_config_from_ui
    gp.update_config_from_ui(
        checkpoint="custom_cp.pt",
        speaker_mode="Custom Voice Clone",
        speaker="geek",
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
        max_chunk_size=150,
    )
    assert gp.config.checkpoint == "custom_cp.pt"
    assert isinstance(gp.config.speaker, CloneSpeaker)
    assert gp.config.speaker.ui_mode == "Custom Voice Clone"
    assert gp.config.speaker.filename == "ref.wav"
    assert gp.config.speaker.text == "ref text"
    assert gp.config.text == "custom text"
    assert gp.config.pause_duration == 0.5
    assert gp.config.temperature == 0.8
    assert gp.config.top_k == 40
    assert gp.config.mbd == MbdMode.Enabled
    assert gp.config.seed_number == 123
    assert gp.config.randomize_seed == RandomMode.Disabled
    assert gp.config.nikud_converter == NikudMode.Disabled
    assert gp.config.max_chunk_size == 150
    # Export & Import settings
    exported_json = gp.export_settings()
    assert os.path.exists(exported_json)
    try:
        # Reset and import
        gp.config.reset()
        assert gp.config.temperature == 1.0
        assert gp.config.max_chunk_size == 100
        imported_tuple = gp.import_settings(exported_json)
        assert len(imported_tuple) == 14
        assert gp.config.temperature == 0.8
        assert gp.config.checkpoint == "custom_cp.pt"
        assert gp.config.max_chunk_size == 150
    finally:
        if os.path.exists(exported_json):
            os.remove(exported_json)
    print("[PASS] GradioPage config integration verified.")


def test_table_printing():
    s = SampleMaker.__new__(SampleMaker)
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


if __name__ == "__main__":
    test_generate_audio_strict_config()
    test_gradio_page_config_helpers()
    test_table_printing()
    print("\n🎉 ALL TESTS IN tests/test_gradio_page.py PASSED SUCCESSFULLY!")
