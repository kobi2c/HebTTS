import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import Config, Format, ChainMode, NoiseFix, split_text
from make_voice import VoiceMaker, VoiceMakerParser


def test_mki_parsing_with_config():
    vm = VoiceMaker()
    base_cfg = Config(speaker="osim", temperature=1.0)
    lines = [
        "[This is a comment]",
        "@{\"speaker\": \"speaker2\", \"temperature\": 0.6}",
        "שלום שורה ראשונה",
        "!{\"speaker\": \"speaker3\"}",
        "שלום שורה שנייה",
    ]
    blocks = vm.parse_mki(lines, base_cfg)
    assert len(blocks) == 2
    # Block 1: incremental update
    b1_cfg, b1_text = blocks[0]
    assert b1_cfg.speaker.id == "speaker2"
    assert b1_cfg.temperature == 0.6
    assert b1_text == "שלום שורה ראשונה"
    # Block 2: reset to base + update
    b2_cfg, b2_text = blocks[1]
    assert b2_cfg.speaker.id == "speaker3"
    assert b2_cfg.temperature == 1.0  # reset back to base_cfg
    assert b2_text == "שלום שורה שנייה"

    # Test split_text with default max_chunk_size=100
    long_hebrew = (
        "היום אנחנו הולכים ללמוד על העולם הקסום של כללי ההתחלקות, "
        "אותם קיצורי דרך נחמדים שמצילים אותנו מהמטלה המייגעת לחלוטין של חילוק ארוך. "
        "האם יצא לכם להביט במספר ענק ולדעת מיד אם אפשר לפצל אותו בשלמות לקבוצות קטנות יותר?"
    )
    chunks = split_text(long_hebrew)
    for c in chunks:
        assert len(c) <= 100, f"Chunk exceeded max_chunk_size=100: {len(c)} chars ({c})"
    assert len(chunks) >= 2

    # Test NoiseFix.fade
    raw_audio = np.ones(2400, dtype=np.float32)  # 100ms at 24kHz
    faded = NoiseFix.fade(raw_audio, sr=24000, fade_in_ms=20.0, fade_out_ms=20.0)
    assert faded[0] == 0.0, f"Expected 0.0 at start, got {faded[0]}"
    assert faded[-1] == 0.0, f"Expected 0.0 at end, got {faded[-1]}"
    assert np.isclose(faded[480], 1.0, atol=1e-3), "Expected mid-body to be 1.0"
    assert np.array_equal(NoiseFix.fade(raw_audio, fade_in_ms=0, fade_out_ms=0), raw_audio)
    assert len(NoiseFix.fade(np.array([], dtype=np.float32))) == 0

    # Test NoiseFix.trim
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


def test_voice_maker_args():
    # 1. Defaults
    vm = VoiceMaker()
    assert vm.input is None
    assert vm.base_config is None
    assert vm.output_dir is None
    assert vm.max_chunk_size is None
    assert vm.fade_ms == 20.0
    assert vm.trim_noise is True
    assert vm.pad_ms == 30.0
    assert vm.clean_existing is False
    assert vm.range is None
    assert vm.range_only is False
    assert vm.format == [Format.WAV]
    assert isinstance(vm.create_parser(), VoiceMakerParser)

    # 2. parse_args with list
    parsed = vm.parse_args([
        "-i", "data/a1.mki",
        "-c", "data/config1.json",
        "-o", "custom_out",
        "--max_chunk_size", "80",
        "--fade_ms", "15.0",
        "--no-trim_noise",
        "--pad_ms", "25.0",
        "--clean_existing",
        "--range", "1-3",
        "--range-only",
        "--format", "both",
    ])
    assert parsed.input == "data/a1.mki"
    assert vm.input == "data/a1.mki"
    assert vm.base_config == "data/config1.json"
    assert vm.output_dir == "custom_out"
    assert vm.max_chunk_size == 80
    assert vm.fade_ms == 15.0
    assert vm.trim_noise is False
    assert vm.pad_ms == 25.0
    assert vm.clean_existing is True
    assert vm.range == "1-3"
    assert vm.range_only is True
    assert vm.format == [Format.WAV, Format.MP3]

    # 3. Direct constructor initialization with List[Format]
    vm2 = VoiceMaker(input="test.mki", max_chunk_size=50, fade_ms=10.0, range_only=True, format=[Format.WAV, Format.MP3])
    assert vm2.input == "test.mki"
    assert vm2.max_chunk_size == 50
    assert vm2.fade_ms == 10.0
    assert vm2.trim_noise is True
    assert vm2.range_only is True
    assert vm2.format == [Format.WAV, Format.MP3]
    # 4. parse_args with existing Namespace
    parser = VoiceMakerParser()
    ns = parser.parse_args(["-i", "ns_input.mki", "--range", "4-6"])
    vm2.parse_args(ns)
    assert vm2.input == "ns_input.mki"
    assert vm2.range == "4-6"
    assert vm2.range_only is False

    # 5. --range-only with direct range string argument
    vm.parse_args(["-i", "test.mki", "--range-only", "2-5"])
    assert vm.range == "2-5"
    assert vm.range_only is True

    # 6. --range_only with underscore variant
    vm.parse_args(["-i", "test.mki", "--range", "3", "--range_only"])
    assert vm.range == "3"
    assert vm.range_only is True

    # 7. --format mp3 sets format to [Format.MP3]
    vm.parse_args(["-i", "test.mki", "--format", "mp3"])
    assert vm.format == [Format.MP3]

    # 8. Direct initialization with ChainMode
    vm3 = VoiceMaker(chain_speaker=ChainMode.Disabled)
    assert vm3.chain_speaker == ChainMode.Disabled
    print("[PASS] VoiceMaker instance fields and parse_args verified.")


def test_voice_maker_range_only_run():
    import tempfile
    import soundfile as sf
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mki_file = tmp_path / "test.mki"
        mki_file.write_text("שורה ראשונה\nשורה שנייה\nשורה שלישית\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        vm = VoiceMaker()
        # Mock synthesis to avoid model weight loading in unit test
        def mock_synth(chunk_text, model=None, **kwargs):
            if model is not None:
                model.last_eos = (195, 425)
            return np.ones(24000, dtype=np.float32) * 0.1  # 1 second of audio at 24kHz

        vm.synthesize_chunk = mock_synth
        vm.page.load_model_if_needed = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()))

        # Run with --range "2" --range-only
        vm.run(["-i", str(mki_file), "--range", "2", "--range-only", "-o", str(out_dir)])

        # Verify only chunk 2 was created
        assert (out_dir / "test_002.wav").exists()
        assert not (out_dir / "test_001.wav").exists()
        assert not (out_dir / "test_003.wav").exists()

        # Verify merged audio contains only chunk 2 (1.0 sec)
        merged_wav, sr = sf.read(str(out_dir / "test.wav"))
        assert sr == 24000
        assert len(merged_wav) == 24000  # exactly 1.0 second, no trailing silence

        # Verify subtitles contain only 1 entry
        srt_content = (out_dir / "test.srt").read_text(encoding="utf-8").strip()
        assert "1\n00:00:00,000 --> 00:00:01,000\nשורה שנייה" in srt_content
        assert "2\n" not in srt_content

        # Clean up output dir for second test
        for f in out_dir.iterdir():
            f.unlink()

        # Run with --range-only "1,3"
        vm.run(["-i", str(mki_file), "--range-only", "1,3", "-o", str(out_dir)])
        assert (out_dir / "test_001.wav").exists()
        assert (out_dir / "test_003.wav").exists()
        assert not (out_dir / "test_002.wav").exists()

        merged_wav2, _ = sf.read(str(out_dir / "test.wav"))
        # 1.0s + 0.3s pause + 1.0s = 2.3s = 55200 samples
        assert len(merged_wav2) == 24000 + int(round(24000 * 0.3)) + 24000

        srt_content2 = (out_dir / "test.srt").read_text(encoding="utf-8")
        assert "שורה ראשונה" in srt_content2
        assert "שורה שלישית" in srt_content2
        assert "שורה שנייה" not in srt_content2

    print("[PASS] VoiceMaker --range-only execution and join verified.")


def test_voice_maker_format_options_run():
    import tempfile
    import soundfile as sf
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mki_file = tmp_path / "fmt_test.mki"
        mki_file.write_text("שורה ראשונה לבדיקת פורמט\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        vm = VoiceMaker()
        def mock_synth(chunk_text, model=None, **kwargs):
            if model is not None:
                model.last_eos = (195, 425)
            return np.ones(24000, dtype=np.float32) * 0.1

        vm.synthesize_chunk = mock_synth
        vm.page.load_model_if_needed = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()))

        # 1. Test --format both
        vm.run(["-i", str(mki_file), "--format", "both", "-o", str(out_dir)])
        assert (out_dir / "fmt_test.wav").exists()
        assert (out_dir / "fmt_test.mp3").exists()
        wav_data, sr_wav = sf.read(str(out_dir / "fmt_test.wav"))
        mp3_data, sr_mp3 = sf.read(str(out_dir / "fmt_test.mp3"))
        assert sr_wav == 24000
        assert sr_mp3 == 24000

        # Clean directory
        for f in out_dir.iterdir():
            f.unlink()

        # 2. Test --format mp3 (only MP3)
        vm.run(["-i", str(mki_file), "--format", "mp3", "-o", str(out_dir)])
        assert (out_dir / "fmt_test.mp3").exists()
        assert not (out_dir / "fmt_test.wav").exists()

        # Clean directory
        for f in out_dir.iterdir():
            f.unlink()

        # 3. Test default (only WAV)
        vm.run(["-i", str(mki_file), "-o", str(out_dir)])
        assert (out_dir / "fmt_test.wav").exists()
        assert not (out_dir / "fmt_test.mp3").exists()

    print("[PASS] VoiceMaker --format output execution verified.")


def test_voice_maker_table_progress_output():
    import io
    import tempfile
    from contextlib import redirect_stdout
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mki_file = tmp_path / "table_test.mki"
        mki_file.write_text("שורה אחת\nשורה שתיים\nשורה שלוש\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        vm = VoiceMaker()
        mock_model = MagicMock()
        mock_model.last_eos = (195, 425)

        def mock_synth(chunk_text, model=None, **kwargs):
            if model is not None:
                model.last_eos = (195, 425)
            return np.ones(24000, dtype=np.float32) * 0.1

        vm.synthesize_chunk = mock_synth
        vm.page.load_model_if_needed = MagicMock(return_value=(mock_model, MagicMock(), MagicMock(), MagicMock()))

        buf = io.StringIO()
        with redirect_stdout(buf):
            vm.run(["-i", str(mki_file), "--range-only", "2", "-o", str(out_dir)])
        out = buf.getvalue()

        # Verify table header
        assert "Chunk" in out
        assert "Status" in out
        assert "Tokens (Prompt->Total)" in out

        # Verify rows
        assert "SKIPPED" in out
        assert "CREATED_S" in out
        assert "195 -> 425 (+230)" in out

        # Verify summary
        assert "Summary: 1 created, 0 reused, 2 skipped (3 total chunks)" in out
        assert mock_model.verbose is False

        # Test text override in make_voice table progress
        for f in out_dir.iterdir():
            f.unlink()
        override_file = out_dir / "table_test_002.input.txt"
        override_file.write_text("שורה שתיים שנדרסה", encoding="utf-8")

        buf_override = io.StringIO()
        with redirect_stdout(buf_override):
            vm.run(["-i", str(mki_file), "--range-only", "2", "-o", str(out_dir)])
        out_override = buf_override.getvalue()

        assert "CREATED_T" in out_override
        assert "Summary: 1 created (T: 1), 0 reused, 2 skipped (3 total chunks)" in out_override

    print("[PASS] VoiceMaker structured table progress and summary verified.")


def test_voice_maker_chain_speaker_execution():
    import tempfile
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mki_file = tmp_path / "chain_test.mki"
        mki_file.write_text("שורה ראשונה\nשורה שנייה\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        vm = VoiceMaker()
        mock_model = MagicMock()
        mock_model.last_eos = (195, 425)

        recorded_prompts = []

        def mock_synth(chunk_text, model=None, prompt_audio=None, prompt_text=None, **kwargs):
            if model is not None:
                model.last_eos = (195, 425)
            recorded_prompts.append((prompt_audio, prompt_text))
            return np.ones(24000, dtype=np.float32) * 0.1

        vm.synthesize_chunk = mock_synth
        vm.page.load_model_if_needed = MagicMock(return_value=(mock_model, MagicMock(), MagicMock(), MagicMock()))

        # 1. Run with default settings (default chain_speaker=ChainMode.Disabled)
        vm.run(["-i", str(mki_file), "-o", str(out_dir)])
        assert len(recorded_prompts) == 2
        # Both chunks use base speaker fallback (prompt_audio is None)
        assert recorded_prompts[0][0] is None
        assert recorded_prompts[1][0] is None

        # 2. Run with explicit chain_speaker=ChainMode.Enabled
        recorded_prompts.clear()
        for f in out_dir.iterdir():
            f.unlink()

        vm_chain = VoiceMaker(chain_speaker=ChainMode.Enabled)
        vm_chain.synthesize_chunk = mock_synth
        vm_chain.page.load_model_if_needed = MagicMock(return_value=(mock_model, MagicMock(), MagicMock(), MagicMock()))
        vm_chain.run(["-i", str(mki_file), "-o", str(out_dir)])
        assert len(recorded_prompts) == 2
        # Chunk 1: base speaker fallback (prompt_audio is None)
        assert recorded_prompts[0][0] is None
        # Chunk 2: chained from chunk 1 output wav & text
        assert recorded_prompts[1][0] == str(out_dir / "chain_test_001.wav")
        assert recorded_prompts[1][1] == "שורה ראשונה"

        # 3. Run with MKI directive enabling chain_speaker
        recorded_prompts.clear()
        for f in out_dir.iterdir():
            f.unlink()

        mki_enabled = tmp_path / "chain_enabled.mki"
        mki_enabled.write_text("@{\"chain_speaker\": 1}\nשורה ראשונה\nשורה שנייה\n", encoding="utf-8")
        vm_default = VoiceMaker()
        vm_default.synthesize_chunk = mock_synth
        vm_default.page.load_model_if_needed = MagicMock(return_value=(mock_model, MagicMock(), MagicMock(), MagicMock()))
        vm_default.run(["-i", str(mki_enabled), "-o", str(out_dir)])
        assert len(recorded_prompts) == 2
        assert recorded_prompts[0][0] is None
        assert recorded_prompts[1][0] == str(out_dir / "chain_enabled_001.wav")
        assert recorded_prompts[1][1] == "שורה ראשונה"

    print("[PASS] VoiceMaker sequential speaker prompt chaining execution verified.")


def test_voice_maker_max_chunk_size():
    import tempfile
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mki_file = tmp_path / "chunk_test.mki"
        long_line = "היום אנחנו הולכים ללמוד על העולם הקסום של כללי ההתחלקות אותם קיצורי דרך שמצילים אותנו"
        mki_file.write_text(f"{long_line}\n", encoding="utf-8")
        cfg_file = tmp_path / "base_cfg.json"
        Config(max_chunk_size=40).saveJson(str(cfg_file))
        out_dir = tmp_path / "out"

        synthesized_chunks = []
        def mock_synth(chunk_text, model=None, **kwargs):
            synthesized_chunks.append(chunk_text)
            return np.ones(24000, dtype=np.float32) * 0.1

        # 1. Inherits max_chunk_size=40 from base_config JSON
        vm = VoiceMaker()
        vm.synthesize_chunk = mock_synth
        vm.page.load_model_if_needed = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()))
        vm.run(["-i", str(mki_file), "-c", str(cfg_file), "-o", str(out_dir)])
        assert len(synthesized_chunks) > 1
        for c in synthesized_chunks:
            assert len(c) <= 40

        # 2. CLI --max_chunk_size overrides base_config
        synthesized_chunks.clear()
        for f in out_dir.iterdir():
            f.unlink()
        vm.run(["-i", str(mki_file), "-c", str(cfg_file), "--max_chunk_size", "200", "-o", str(out_dir)])
        assert len(synthesized_chunks) == 1
        assert synthesized_chunks[0] == long_line

    print("[PASS] VoiceMaker max_chunk_size inheritance and CLI override verified.")


if __name__ == "__main__":
    test_mki_parsing_with_config()
    test_voice_maker_args()
    test_voice_maker_range_only_run()
    test_voice_maker_format_options_run()
    test_voice_maker_table_progress_output()
    test_voice_maker_chain_speaker_execution()
    test_voice_maker_max_chunk_size()
    print("\n🎉 ALL TESTS IN tests/test_make_voice.py PASSED SUCCESSFULLY!")
