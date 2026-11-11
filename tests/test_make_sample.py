import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

import numpy as np
import torch


from config import Format
from make_sample import SampleMaker, SampleMakerParser


def test_sampler_args():
    # 1. Default constructor fields
    s = SampleMaker()
    assert s.tokens_file == SampleMaker.DEFAULT_TOKENS_FILE
    assert s.vocab_file == SampleMaker.DEFAULT_VOCAB_FILE
    assert s.speaker == "1"
    assert s.checkpoint == "1"
    assert s.input_file is None
    assert s.text is None
    assert s.clone_wav is None
    assert s.clone_txt is None
    assert s.output is None
    assert s.format == [Format.WAV]
    assert s.top_k == 50
    assert s.temperature == 1.0
    assert s.mbd is False
    assert s.nikud_mode is True
    assert s.speaker_list is False
    assert s.checkpoint_list is False
    assert isinstance(s.create_parser(), SampleMakerParser)

    # 2. parse_args with list
    parsed = s.parse_args([
        "--speaker", "osim",
        "--checkpoint", "1",
        "--text", "שלום עולם",
        "--clone-audio", "ref.mp3",
        "--clone-txt", "ref.txt",
        "--output", "custom_out.mp3",
        "--format", "mp3",
        "--top-k", "40",
        "--temperature", "0.7",
        "--mbd",
        "--no-nikud-mode",
    ])
    assert parsed.speaker == "osim"
    assert s.speaker == "osim"
    assert s.checkpoint == "1"
    assert s.text == "שלום עולם"
    assert s.clone_wav == "ref.mp3"
    assert s.clone_txt == "ref.txt"
    assert s.output == "custom_out.mp3"
    assert s.format == [Format.MP3]
    assert s.top_k == 40
    assert s.temperature == 0.7
    assert s.mbd is True
    assert s.nikud_mode is False

    # 3. Direct constructor initialization with fields
    s2 = SampleMaker(
        speaker="osim",
        checkpoint="2",
        text="בדיקה ישירה",
        output="out.wav",
        format=[Format.MP3],
        top_k=25,
        temperature=0.9,
        mbd=True,
        nikud_mode=False,
    )
    assert s2.speaker == "osim"
    assert s2.checkpoint == "2"
    assert s2.text == "בדיקה ישירה"
    assert s2.output == "out.wav"
    assert s2.format == [Format.MP3]
    assert s2.top_k == 25
    assert s2.temperature == 0.9
    assert s2.mbd is True
    assert s2.nikud_mode is False

    # 4. parse_args with existing Namespace
    parser = SampleMakerParser()
    ns = parser.parse_args(["--speaker", "1", "--text", "בדיקת Namespace"])
    s2.parse_args(ns)
    assert s2.speaker == "1"
    assert s2.text == "בדיקת Namespace"

    # 5. --format mp3 sets format to [Format.MP3]
    s.parse_args(["--format", "mp3"])
    assert s.format == [Format.MP3]

    # 6. --format both sets format to [Format.WAV, Format.MP3]
    s.parse_args(["--format", "both"])
    assert s.format == [Format.WAV, Format.MP3]

    print("[PASS] SampleMaker CLI, instance fields, and parse_args verified.")


def test_sampler_list_commands():
    s_spk = SampleMaker(speaker_list=True)
    f = io.StringIO()
    with redirect_stdout(f):
        res = s_spk.run()
    assert res is None
    assert "Available speakers:" in f.getvalue()

    s_cp = SampleMaker(checkpoint_list=True)
    f = io.StringIO()
    with redirect_stdout(f):
        res = s_cp.run()
    assert res is None
    assert "Available checkpoints:" in f.getvalue()

    print("[PASS] SampleMaker --speaker-list and --checkpoint-list execution verified.")


def test_sampler_run_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        out_wav = tmp_path / "test_out.wav"
        out_mp3 = tmp_path / "test_out.mp3"

        s = SampleMaker(speaker="1", text="שלום עולם", output=out_wav)
        s.engine.create = MagicMock(return_value=np.zeros(24000, dtype=np.float32))

        # 1. Run with instance fields (WAV output)
        result = s.run()
        assert result == out_wav
        assert out_wav.exists()
        assert s.engine.create.called

        # 2. Run with MP3 format via parse_args in run()
        result_mp3 = s.run(["--text", "שלום שוב", "--format", "mp3", "--output", str(out_mp3)])
        assert result_mp3 == out_mp3
        assert out_mp3.exists()

        # 3. Run with both formats via --format both
        both_base = tmp_path / "both_out"
        s.run(["--text", "שניהם", "--format", "both", "--output", str(both_base)])
        assert (tmp_path / "both_out.wav").exists()
        assert (tmp_path / "both_out.mp3").exists()

        # 4. Run with custom pluggable engine
        from engine import VoiceEngine
        class MockCustomEngine(VoiceEngine):
            def create(self, *args, **kwargs):
                return np.zeros(16000, dtype=np.float32)
            @property
            def sample_rate(self):
                return 16000

        custom_eng = MockCustomEngine()
        s_custom = SampleMaker(engine=custom_eng, speaker="1", text="בדיקת מנוע", output=tmp_path / "cust.wav")
        assert s_custom.engine is custom_eng
        cust_res = s_custom.run()
        assert cust_res == tmp_path / "cust.wav"
        assert (tmp_path / "cust.wav").exists()

    print("[PASS] SampleMaker.run() execution without args dependency verified.")


if __name__ == "__main__":
    test_sampler_args()
    test_sampler_list_commands()
    test_sampler_run_execution()
    print("\n🎉 ALL TESTS IN tests/test_make_sample.py PASSED SUCCESSFULLY!")
