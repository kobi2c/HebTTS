import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
from unittest.mock import MagicMock

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

import numpy as np
import soundfile as sf
import torch

from config import Config, Speaker, LocalSpeaker, SpeakerSample, SeedMode
from engine import (
    EngineInfo,
    VoiceEngine,
    HebTTSEngine,
    get_engine,
    register_engine,
    unregister_engine,
    list_engines,
    get_engine_class,
    get_engine_info,
)
from make_voice import VoiceMaker


class DummyBeepEngine(VoiceEngine):
    """Custom pluggable engine for testing alternative synthesis models."""

    info = EngineInfo(
        id="dummy_beep",
        name="Dummy Beep Engine",
        sample_rate=16000,
        languages=["he"],
    )

    def __init__(self, sample_rate: int = 16000) -> None:
        if sample_rate != 16000:
            self.info = EngineInfo(
                id="dummy_beep",
                name="Dummy Beep Engine",
                sample_rate=sample_rate,
                languages=["he"],
            )
        self.calls = []

    @property
    def last_tokens_str(self) -> str:
        return "dummy: 10 -> 20 (+10)"

    def create_panel(self) -> Any:
        return MagicMock()

    def synthesize_chunk(
        self,
        text: str,
        speaker: Speaker,
        sample: Optional[SpeakerSample] = None,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        self.calls.append({
            "speaker": speaker,
            "sample": sample,
            "config": config,
            "text": text,
        })
        # Produce 0.5s of audio at sample_rate
        num_samples = int(self.info.sample_rate * 0.5)
        return np.ones(num_samples, dtype=np.float32) * 0.2


def test_voice_engine_abc():
    # 1. Cannot instantiate abstract VoiceEngine directly
    try:
        VoiceEngine()
        assert False, "Should have raised TypeError when instantiating VoiceEngine ABC"
    except TypeError:
        pass

    # 2. Subclass missing synthesize_chunk() cannot be instantiated
    class IncompleteEngine(VoiceEngine):
        pass

    try:
        IncompleteEngine()
        assert False, "Should have raised TypeError for IncompleteEngine"
    except TypeError:
        pass

    # 3. Seed management on concrete subclass
    dummy = DummyBeepEngine()
    assert dummy.apply_seed(42, SeedMode.Disabled) == 42
    s = dummy.apply_seed(None, SeedMode.Enabled)
    assert isinstance(s, int) and s >= 0

    # 4. Engine metadata & inspection
    assert dummy.info.id == "dummy_beep"
    assert dummy.info.name == "Dummy Beep Engine"
    assert dummy.is_loaded is False
    assert dummy.loaded_checkpoint is None
    info = dummy.live_info()
    assert isinstance(info, EngineInfo)
    assert info.id == "dummy_beep"
    assert info.name == "Dummy Beep Engine"
    assert info.sample_rate == 16000
    assert isinstance(info.to_dict(), dict)
    assert info.to_dict()["name"] == "Dummy Beep Engine"

    # 5. Lifecycle methods
    dummy.unload_model()
    dummy.clear_cache()
    dummy.clear_last_tokens()
    assert dummy.create_panel() is not None

    print("[PASS] VoiceEngine ABC contract verified.")


def test_get_engine_factory():
    # 1. Default -> HebTTSEngine
    eng_def = get_engine()
    assert isinstance(eng_def, HebTTSEngine)
    assert eng_def.info.sample_rate == 24000

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


def test_engine_registry():
    engines = list_engines()
    assert "hebtts" in engines
    assert "zonos" in engines

    heb_info = get_engine_info("hebtts")
    assert isinstance(heb_info, EngineInfo)
    assert heb_info.id == "hebtts"
    assert heb_info.sample_rate == 24000

    zonos_info = get_engine_info("zonos")
    assert isinstance(zonos_info, EngineInfo)
    assert zonos_info.id == "zonos"
    assert zonos_info.sample_rate == 44100

    # Custom engine registration
    register_engine(["mock_custom", "custom_alias"], DummyBeepEngine)
    assert "mock_custom" in list_engines()
    assert get_engine("mock_custom").info.sample_rate == 16000
    assert get_engine("custom_alias").info.sample_rate == 16000

    # Unregister
    unregister_engine("mock_custom")
    assert "mock_custom" not in list_engines()
    try:
        get_engine("mock_custom")
        assert False, "Should raise ValueError after unregister"
    except ValueError:
        pass

    print("[PASS] Dynamic engine registry verified.")


def test_heb_tts_engine_create():
    mock_model = MagicMock()
    mock_model.last_eos = (195, 425)

    mock_tok = MagicMock()
    mock_encoded = torch.zeros((1, 8, 50), dtype=torch.long)
    mock_tok.encode.return_value = [[mock_encoded]]
    mock_tok.sample_rate = 24000
    mock_tok.channels = 1

    engine = HebTTSEngine()
    engine.audio_tokenizer = mock_tok
    engine.load_model = MagicMock(return_value=(mock_model, mock_tok, MagicMock(), MagicMock()))

    def fake_synthesize(*args, **kwargs):
        mock_model.last_eos = (195, 425)
        engine._last_tokens_str = "195 -> 425 (+230)"
        return np.ones(2400, dtype=np.float32) * 0.5

    engine.synthesize_chunk = MagicMock(side_effect=fake_synthesize)

    assert engine.info.id == "hebtts"
    assert engine.info.name == "HebTTS (VALL-E)"
    assert engine.info.sample_rate == 24000
    assert engine.last_tokens_str is None
    assert engine.device is not None
    assert engine.device_name is not None
    assert engine.apply_seed(999, SeedMode.Disabled) == 999
    rnd_s = engine.apply_seed(None, SeedMode.Enabled)
    assert isinstance(rnd_s, int) and rnd_s >= 0

    info = engine.live_info()
    assert isinstance(info, EngineInfo)
    assert info.id == "hebtts"
    assert info.sample_rate == 24000
    assert "he" in info.languages
    assert isinstance(info.to_dict(), dict)

    spk = LocalSpeaker(id="osim", filename="osim.wav", text="שלום עולם")
    cfg = Config(
        speaker=spk,
        temperature=0.8,
        top_k=40,
        seed=777,
        seed_mode=SeedMode.Disabled,
        fade_ms=0.0,
        trim_noise=False,
    )

    progress_events = []
    def on_progress(idx, total, desc):
        progress_events.append((idx, total, desc))

    audio = engine.create(
        speaker=spk,
        sample=SpeakerSample(id="osim", path="speakers/osim.wav", text="שלום עולם"),
        config=cfg,
        text="בדיקת דיבור שורה 1\nבדיקת דיבור שורה 2",
        progress_callback=on_progress,
    )

    assert isinstance(audio, np.ndarray)
    assert len(progress_events) == 2
    assert engine.last_tokens_str == "195 -> 425 (+230)"
    assert engine.synthesize_chunk.call_count == 2
    assert cfg.seed == 777

    # SeedMode.Enabled generates random seed in cfg.seed
    cfg_rnd = Config(speaker=spk, seed_mode=SeedMode.Enabled)
    engine.create(speaker=spk, sample=spk.sample(), config=cfg_rnd, text="בדיקה")
    assert isinstance(cfg_rnd.seed, int) and cfg_rnd.seed >= 0

    # Test unload_model & clear_cache
    engine.unload_model()
    assert engine.is_loaded is False
    assert engine.loaded_checkpoint is None
    engine.clear_cache()
    assert engine.last_tokens_str is None

    print("[PASS] HebTTSEngine create and token stats verified.")


def test_heb_tts_engine_synthesize_chunk():
    engine = HebTTSEngine()
    engine._device = torch.device("cpu")

    mock_model = MagicMock()
    mock_model.inference.return_value = torch.zeros((1, 8, 50))
    engine.model = mock_model

    mock_sample = MagicMock(shape=(1, 24000))
    mock_sample.cpu.return_value = mock_sample
    mock_sample.squeeze.return_value.cpu.return_value.numpy.return_value = np.zeros(24000, dtype=np.float32)
    mock_audio_tok = MagicMock()
    mock_audio_tok.decode.return_value = [mock_sample]
    mock_audio_tok.mbd_decode.return_value = [mock_sample]
    mock_audio_tok.mbd = MagicMock()
    engine.audio_tokenizer = mock_audio_tok

    mock_text_collater = MagicMock()
    mock_text_collater.token2idx = {"tok1": 1, "tok2": 2}
    mock_text_collater.side_effect = lambda seqs: (torch.zeros((1, 5)), torch.tensor([5]))
    engine.text_collater = mock_text_collater

    mock_alef_bert = MagicMock()
    mock_alef_bert._tokenize.return_value = ["tok1", "tok2"]
    engine.alef_bert_tokenizer = mock_alef_bert

    spk = LocalSpeaker(id="osim", filename="osim.wav", text="פרומפט")
    sample = SpeakerSample(id="osim", path="osim.wav", text="פרומפט")
    engine.prepare_speaker = MagicMock(return_value=torch.zeros((1, 8, 20)))

    # 1. Non-MBD synthesis: audio_tokenizer.decode called with [(frames, None)]
    out = engine.synthesize_chunk(
        text="שלום עולם",
        speaker=spk,
        sample=sample,
        config=Config(mbd=0),
    )
    assert isinstance(out, np.ndarray)
    assert mock_audio_tok.decode.called
    decode_call_arg = mock_audio_tok.decode.call_args[0][0]
    assert isinstance(decode_call_arg, list)
    assert len(decode_call_arg) == 1
    assert isinstance(decode_call_arg[0], tuple)
    assert decode_call_arg[0][1] is None, "Second element of decoded frame must be None"

    # 2. MBD synthesis: audio_tokenizer.mbd_decode called
    mock_audio_tok.decode.reset_mock()
    out_mbd = engine.synthesize_chunk(
        text="שלום עולם",
        speaker=spk,
        sample=sample,
        config=Config(mbd=1),
    )
    assert isinstance(out_mbd, np.ndarray)
    assert mock_audio_tok.mbd_decode.called
    assert not mock_audio_tok.decode.called

    print("[PASS] HebTTSEngine synthesize_chunk device contracts and vocoder routing verified.")


def test_voice_engine_lifecycle():
    engine = DummyBeepEngine(sample_rate=24000)

    # 1. Base prepare_text with str and list inputs
    res_str = engine.prepare_text("  line 1  \n\n line 2 \n  ")
    assert res_str == ["line 1", "line 2"]

    res_list = engine.prepare_text(["line A\nline B", "  line C  "])
    assert res_list == ["line A", "line B", "line C"]

    assert engine.prepare_text("") == []

    # split_text directly and via prepare_text respecting max_chunk_size
    parts_split = VoiceEngine.split_text("משפט ראשון. משפט שני.", max_chunk_size=15)
    assert parts_split == ["משפט ראשון.", "משפט שני."]

    parts_prep = engine.prepare_text("משפט ראשון. משפט שני.", Config(max_chunk_size=15))
    assert parts_prep == ["משפט ראשון.", "משפט שני."]

    # 2. HebTTSEngine prepare_text with nikud conversion via preprocess_text
    heb_engine = HebTTSEngine()
    cfg_nikud_on = Config(nikud_mode=1)
    cfg_nikud_off = Config(nikud_mode=0)

    # Mock nikud_maker.convert
    heb_engine.nikud_maker.convert = MagicMock(side_effect=lambda s: f"converted_{s}")

    lines_on = heb_engine.prepare_text("שלום", cfg_nikud_on)
    assert lines_on == ["converted_שלום"]

    lines_off = heb_engine.prepare_text("שלום", cfg_nikud_off)
    assert lines_off == ["שלום"]

    # 3. Base post_create with peak clipping and post-processing
    audio_clipping = np.array([0.0, 2.0, -2.0, 0.5], dtype=np.float32)
    cfg_post = Config(fade_ms=0.0, trim_noise=False)
    processed = engine.post_create(audio_clipping, cfg_post)
    assert np.isclose(np.abs(processed).max(), 1.0)

    # post_create with empty array returns empty
    assert len(engine.post_create(np.zeros(0, dtype=np.float32), cfg_post)) == 0

    print("[PASS] VoiceEngine prepare_text and post_create lifecycle verified.")


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
    test_engine_registry()
    test_heb_tts_engine_create()
    test_heb_tts_engine_synthesize_chunk()
    test_voice_engine_lifecycle()
    test_voice_maker_with_custom_engine()
    print("\n🎉 ALL TESTS IN tests/test_engine.py PASSED SUCCESSFULLY!")
