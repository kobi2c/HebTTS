import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

import numpy as np
import torch

from config import Config, LocalSpeaker, SeedMode
from engine import VoiceEngine, get_engine
from engine.zonos import ZonosEngine


def test_zonos_engine_contract():
    eng = ZonosEngine()
    assert isinstance(eng, VoiceEngine)
    assert eng.sample_rate == 44100
    assert eng.default_model_choice == "notmax123/Zonos-Hebrew"
    assert eng.last_tokens_str is None
    assert eng.cached_model_choice is None

    # Custom model choice & device
    eng_custom = ZonosEngine(model_choice="Zyphra/Zonos-v0.1-transformer", device="cpu")
    assert eng_custom.default_model_choice == "Zyphra/Zonos-v0.1-transformer"
    assert eng_custom.sample_rate == 44100
    assert "cpu" in str(eng_custom.device).lower()

    print("[PASS] ZonosEngine ABC contract and properties verified.")


def test_zonos_engine_factory():
    eng1 = get_engine("zonos")
    assert isinstance(eng1, ZonosEngine)
    assert eng1.sample_rate == 44100

    eng2 = get_engine("zonos_hebrew")
    assert isinstance(eng2, ZonosEngine)

    eng3 = get_engine("zonos_engine")
    assert isinstance(eng3, ZonosEngine)

    # Lazy attribute export from engine
    import engine
    assert hasattr(engine, "ZonosEngine")
    assert engine.ZonosEngine is ZonosEngine

    print("[PASS] ZonosEngine factory integration and lazy export verified.")


def test_zonos_engine_create_flow():
    eng = ZonosEngine(device="cpu")

    # Mock Zonos model
    mock_model = MagicMock()
    mock_spk_emb = torch.ones((1, 128), dtype=torch.float32)
    mock_model.make_speaker_embedding.return_value = mock_spk_emb
    mock_model.prepare_conditioning.return_value = MagicMock()
    mock_model.generate.return_value = torch.ones((1, 8, 10), dtype=torch.long)
    # autoencoder decode returns waveform of 44100 samples (1 sec)
    mock_wav = MagicMock()
    mock_wav.dim.return_value = 1
    mock_wav.size.return_value = 44100
    mock_wav.cpu.return_value.detach.return_value.squeeze.return_value.numpy.return_value = np.ones(44100, dtype=np.float32) * 0.5
    mock_model.autoencoder.decode.return_value = mock_wav

    eng.model = mock_model
    eng.cached_model_choice = "notmax123/Zonos-Hebrew"

    # Mock speaker with temporary prompt file
    spk = LocalSpeaker(id="test_spk", filename="test.wav", text="פרומפט בדיקה")
    spk._resolved_path = Path("test.wav")

    # Patch os.path.exists and sf.read for speaker prompt
    with patch("os.path.exists", return_value=True), \
         patch("soundfile.read", return_value=(np.zeros(44100, dtype=np.float32), 44100)):

        cfg = Config(
            seed=123,
            seed_mode=SeedMode.Disabled,
            pause_duration=0.2,
            model_choice="notmax123/Zonos-Hebrew",
        )

        progress_calls = []
        def on_prog(idx, tot, desc):
            progress_calls.append((idx, tot, desc))

        # Synthesize 2 lines
        text = "שלום עולם.\nזהו מודל זונוס."
        audio = eng.create(
            speaker=spk,
            config=cfg,
            text=text,
            progress_callback=on_prog,
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert eng.last_tokens_str is not None
        assert "Zonos [notmax123/Zonos-Hebrew]" in eng.last_tokens_str
        assert len(progress_calls) == 2

        # Verify speaker embedding was cached (only called once for 2 lines)
        assert mock_model.make_speaker_embedding.call_count == 1

        # Second create call with same speaker should reuse cached embedding
        eng.create(speaker=spk, config=cfg, text="שורה נוספת")
        assert mock_model.make_speaker_embedding.call_count == 1

        # Empty text input returns empty array
        empty_res = eng.create(speaker=spk, config=cfg, text="")
        assert len(empty_res) == 0

    print("[PASS] ZonosEngine create synthesis workflow and caching verified.")


if __name__ == "__main__":
    test_zonos_engine_contract()
    test_zonos_engine_factory()
    test_zonos_engine_create_flow()
    print("\n🎉 ALL TESTS IN tests/test_zonos_engine.py PASSED SUCCESSFULLY!")
