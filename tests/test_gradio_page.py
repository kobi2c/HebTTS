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

import gradio as gr
from config import (
    Config,
    SpeakerMode,
    Speaker,
    LocalSpeaker,
    CloneSpeaker,
    MbdMode,
    SeedMode,
    NikudMode,
    DEFAULT_TEXT,
    CheckpointStore,
    SpeakerStore,
)
from GradioPage import GradioPage, HebTTSPanel, ZonosPanel, ENGINE_PANELS
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
    assert gp.config.engine == "hebtts"

    # Check default UI values
    ui_vals = gp.get_ui_values_from_config()
    assert len(ui_vals) == 41

    # Common values
    assert ui_vals[0] == "hebtts"
    assert ui_vals[1] == "Local Speaker"
    assert ui_vals[2] in ("osim", "1")
    assert ui_vals[3] is None
    assert ui_vals[4] == ""
    assert ui_vals[5] == DEFAULT_TEXT

    # HebTTS values (9)
    assert ui_vals[6] == gp.config.checkpoint
    assert ui_vals[7] == 0.3
    assert ui_vals[8] == 1.0
    assert ui_vals[9] == 50
    assert ui_vals[10] == 0  # int(MbdMode.Disabled)
    assert ui_vals[11] == 420
    assert ui_vals[12] == 1  # int(SeedMode.Enabled)
    assert ui_vals[13] == 1  # int(NikudMode.Enabled)
    assert ui_vals[14] == 100

    # Zonos values (19)
    assert ui_vals[15] == "notmax123/Zonos-Hebrew"
    assert ui_vals[16] == "he"
    assert ui_vals[17] == 2.0
    assert ui_vals[18] == 15.0
    assert ui_vals[19] == 45.0
    assert ui_vals[20] == 24000.0
    assert ui_vals[21] == 4.0
    assert ui_vals[22] == 0.78
    assert ui_vals[23] is False
    assert ui_vals[24] == "Neutral / Automatic (Unconditional)"
    assert ui_vals[25:33] == [0.0] * 8
    assert ui_vals[33] == 0.1
    assert ui_vals[34] == 0.0
    assert ui_vals[35] == 0
    assert ui_vals[36] == 0.0
    assert ui_vals[37] == 0.0
    assert ui_vals[38] == 0.0
    assert ui_vals[39] == 420
    assert ui_vals[40] == 1

    # Test update_config_from_ui via kwargs
    gp.update_config_from_ui(
        engine="zonos",
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
        seed=123,
        seed_mode=0,
        nikud_mode=0,
        max_chunk_size=150,
        model_choice="Zyphra/Zonos-v0.1-transformer",
        language="en-us",
        speaking_rate=18.0,
        cfg_scale=3.0,
        emotions=[0.2] * 8,
    )
    assert gp.config.engine == "zonos"
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
    assert gp.config.seed == 123
    assert gp.config.seed_mode == SeedMode.Disabled
    assert gp.config.nikud_mode == NikudMode.Disabled
    assert gp.config.max_chunk_size == 150
    assert gp.config.model_choice == "Zyphra/Zonos-v0.1-transformer"
    assert gp.config.language == "en-us"
    assert gp.config.speaking_rate == 18.0
    assert gp.config.cfg_scale == 3.0
    assert gp.config.emotions == [0.2] * 8

    # Export & Import settings
    exported_json = gp.export_settings()
    assert os.path.exists(exported_json)
    try:
        # Reset and import
        gp.config.reset()
        assert gp.config.engine == "hebtts"
        assert gp.config.temperature == 1.0
        assert gp.config.max_chunk_size == 100
        assert gp.config.model_choice == "notmax123/Zonos-Hebrew"

        imported_tuple = gp.import_settings(exported_json)
        assert len(imported_tuple) == 41 + 8
        assert gp.config.engine == "zonos"
        assert gp.config.temperature == 0.8
        assert gp.config.checkpoint == "custom_cp.pt"
        assert gp.config.max_chunk_size == 150
        assert gp.config.model_choice == "Zyphra/Zonos-v0.1-transformer"
        assert gp.config.speaking_rate == 18.0
        assert gp.config.emotions == [0.2] * 8
    finally:
        if os.path.exists(exported_json):
            os.remove(exported_json)

    # Test update_config_from_ui with positional args
    new_vals = list(gp.get_ui_values_from_config())
    new_vals[0] = "hebtts"
    new_vals[5] = "positional text test"
    new_vals[8] = 0.65  # temperature
    gp.update_config_from_ui(*new_vals)
    assert gp.config.engine == "hebtts"
    assert gp.config.text == "positional text test"
    assert gp.config.temperature == 0.65

    print("[PASS] GradioPage config integration verified.")


def test_gradio_page_build_interface():
    gp = GradioPage()
    demo = gp.build_interface()
    assert demo is not None
    if isinstance(gr.Blocks, type):
        assert isinstance(demo, gr.Blocks)
    assert "hebtts" in gp.panels
    assert "zonos" in gp.panels
    assert len(gp.panels["hebtts"].get_components()) == 9
    assert len(gp.panels["zonos"].get_components()) == 26

    # Verify badge markdown
    heb_badge = gp.get_engine_badge_markdown("hebtts")
    assert "HebTTS (VALL-E)" in heb_badge
    assert "24,000 Hz" in heb_badge

    zonos_badge = gp.get_engine_badge_markdown("zonos")
    assert "Zonos (Zyphra)" in zonos_badge
    assert "44,100 Hz" in zonos_badge
    print("[PASS] GradioPage.build_interface and panels verified.")


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
    test_gradio_page_build_interface()
    test_table_printing()
    print("\n🎉 ALL TESTS IN tests/test_gradio_page.py PASSED SUCCESSFULLY!")
