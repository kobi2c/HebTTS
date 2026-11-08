import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import Config, NoiseFix
from make_voice import VoiceMaker, VoiceMakerParser, split_text


def test_mki_parsing_with_config():
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
    long_hebrew = (
        "היום אנחנו הולכים ללמוד על העולם הקסום של כללי ההתחלקות, "
        "אותם קיצורי דרך נחמדים שמצילים אותנו מהמטלה המייגעת לחלוטין של חילוק ארוך. "
        "האם יצא לכם להביט במספר ענק ולדעת מיד אם אפשר לפצל אותו בשלמות לקבוצות קטנות יותר?"
    )
    chunks = split_text(long_hebrew)
    for c in chunks:
        assert len(c) <= 100, f"Chunk exceeded max_len=100: {len(c)} chars ({c})"
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
    assert vm.max_len == 100
    assert vm.fade_ms == 20.0
    assert vm.trim_noise is True
    assert vm.pad_ms == 30.0
    assert vm.clean_existing is False
    assert vm.range is None
    assert isinstance(vm.create_parser(), VoiceMakerParser)

    # 2. parse_args with list
    parsed = vm.parse_args([
        "-i", "data/a1.mki",
        "-c", "data/config1.json",
        "-o", "custom_out",
        "--max_len", "80",
        "--fade_ms", "15.0",
        "--no-trim_noise",
        "--pad_ms", "25.0",
        "--clean_existing",
        "--range", "1-3",
    ])
    assert parsed.input == "data/a1.mki"
    assert vm.input == "data/a1.mki"
    assert vm.base_config == "data/config1.json"
    assert vm.output_dir == "custom_out"
    assert vm.max_len == 80
    assert vm.fade_ms == 15.0
    assert vm.trim_noise is False
    assert vm.pad_ms == 25.0
    assert vm.clean_existing is True
    assert vm.range == "1-3"

    # 3. Direct constructor initialization
    vm2 = VoiceMaker(input="test.mki", max_len=50, fade_ms=10.0)
    assert vm2.input == "test.mki"
    assert vm2.max_len == 50
    assert vm2.fade_ms == 10.0
    assert vm2.trim_noise is True

    # 4. parse_args with existing Namespace
    parser = VoiceMakerParser()
    ns = parser.parse_args(["-i", "ns_input.mki", "--range", "4-6"])
    vm2.parse_args(ns)
    assert vm2.input == "ns_input.mki"
    assert vm2.range == "4-6"
    print("[PASS] VoiceMaker instance fields and parse_args verified.")


if __name__ == "__main__":
    test_mki_parsing_with_config()
    test_voice_maker_args()
    print("\n🎉 ALL TESTS IN tests/test_make_voice.py PASSED SUCCESSFULLY!")
