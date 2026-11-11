import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

import numpy as np
import soundfile as sf
import torch

from config import Config, Speaker, LocalSpeaker, SpeakerSample
from engine import VoiceEngine, HebTTSEngine, get_engine
from make_voice import VoiceMaker


class DummyBeepEngine(VoiceEngine):
    """Custom pluggable engine for testing alternative synthesis models."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sr = sample_rate
        self.calls = []

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def last_tokens_str(self) -> str:
        return "dummy: 10 -> 20 (+10)"

    def create(
        self,
        speaker: Speaker,
        text: str,
        config: Config = None,
        prompt_sample: SpeakerSample = None,
        fade_ms: float = 20.0,
        trim_noise: bool = True,
        pad_ms: float = 30.0,
        **kwargs,
    ) -> np.ndarray:
        self.calls.append({
            "speaker": speaker,
            "text": text,
            "config": config,
            "prompt_sample": prompt_sample,
        })
        # Produce 0.5s of audio at 16kHz
        num_samples = int(self._sr * 0.5)
        return np.ones(num_samples, dtype=np.float32) * 0.2


def test_voice_engine_abc():
    # 1. Cannot instantiate abstract VoiceEngine directly
    try:
        VoiceEngine()
        assert False, "Should have raised TypeError when instantiating VoiceEngine ABC"
    except TypeError:
        pass

    # 2. Subclass missing create() cannot be instantiated
    class IncompleteEngine(VoiceEngine):
        pass

    try:
        IncompleteEngine()
        assert False, "Should have raised TypeError for IncompleteEngine"
    except TypeError:
        pass

    print("[PASS] VoiceEngine ABC contract verified.")


def test_get_engine_factory():
    # 1. Default -> HebTTSEngine
    eng_def = get_engine()
    assert isinstance(eng_def, HebTTSEngine)
    assert eng_def.sample_rate == 24000

    # 2. String names
    assert isinstance(get_engine("hebtts"), HebTTSEngine)
    assert isinstance(get_engine("heb_tts"), HebTTSEngine)
    assert isinstance(get_engine("valle"), HebTTSEngine)
    assert isinstance(get_engine("valle-hebrew"), HebTTSEngine)

    # 3. Existing instance pass-through
    dummy = DummyBeepEngine()
    assert get_engine(dummy) is dummy

    # 4. Unknown string name
    try:
        get_engine("non_existent_engine")
        assert False, "Should have raised ValueError on unknown engine name"
    except ValueError as e:
        assert "Unknown VoiceEngine name" in str(e)

    # 5. Invalid type
    try:
        get_engine(12345)
        assert False, "Should have raised TypeError on non-string/non-engine"
    except TypeError as e:
        assert "Expected VoiceEngine or str" in str(e)

    print("[PASS] get_engine factory verified.")


def test_heb_tts_engine_create():
    mock_model = MagicMock()
    mock_model.last_eos = (195, 425)

    mock_tensor = MagicMock()
    mock_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = np.ones(2400, dtype=np.float32) * 0.5
    mock_tensor.shape = (1, 2400)

    engine = HebTTSEngine()
    engine.load_model = MagicMock(return_value=(mock_model, MagicMock(), MagicMock(), MagicMock()))

    def fake_synthesize(*args, **kwargs):
        mock_model.last_eos = (195, 425)
        return mock_tensor

    engine.synthesize_line = MagicMock(side_effect=fake_synthesize)

    assert engine.sample_rate == 24000
    assert engine.last_tokens_str is None
    assert engine.device is not None
    assert engine.device_name is not None

    spk = LocalSpeaker(id="osim", filename="osim.wav", text="שלום עולם")
    cfg = Config(speaker=spk, temperature=0.8, top_k=40)

    progress_events = []
    def on_progress(idx, total, desc):
        progress_events.append((idx, total, desc))

    audio = engine.create(
        speaker=spk,
        text="בדיקת דיבור שורה 1\nבדיקת דיבור שורה 2",
        config=cfg,
        prompt_sample=SpeakerSample(id="osim", path="speakers/osim.wav", text="שלום עולם"),
        fade_ms=0.0,
        trim_noise=False,
        progress_callback=on_progress,
    )

    assert isinstance(audio, np.ndarray)
    assert len(progress_events) == 2
    assert engine.last_tokens_str == "195 -> 425 (+230)"
    assert engine.synthesize_line.call_count == 2

    print("[PASS] HebTTSEngine create and token stats verified.")


def test_voice_maker_with_custom_engine():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mki_file = tmp_path / "custom_engine_test.mki"
        mki_file.write_text("שורה ראשונה במנוע מותאם אישית\nשורה שנייה במנוע מותאם אישית\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        custom_engine = DummyBeepEngine(sample_rate=16000)
        vm = VoiceMaker(engine=custom_engine)
        assert vm.engine is custom_engine

        vm.run(["-i", str(mki_file), "-o", str(out_dir)])

        # Verify output audio was saved at 16000 Hz
        merged_file = out_dir / "custom_engine_test.wav"
        assert merged_file.exists()
        audio_data, sr = sf.read(str(merged_file))
        assert sr == 16000
        assert len(custom_engine.calls) == 2

        # Verify chunk token string
        assert custom_engine.last_tokens_str == "dummy: 10 -> 20 (+10)"

    print("[PASS] VoiceMaker with custom pluggable VoiceEngine verified.")


if __name__ == "__main__":
    test_voice_engine_abc()
    test_get_engine_factory()
    test_heb_tts_engine_create()
    test_voice_maker_with_custom_engine()
    print("\n🎉 ALL TESTS IN tests/test_engine.py PASSED SUCCESSFULLY!")
