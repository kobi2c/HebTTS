import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

# Silence torchvision image extension warning since TTS does not use vision IO
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Silence deprecated weight_norm FutureWarnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*weight_norm.*")

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Gracefully mock container-only dependencies if running in a lightweight host environment
for mod in [
    "omegaconf",
    "soundfile",
    "gradio",
    "torch",
    "torchaudio",
    "valle",
    "valle.data",
    "valle.data.collation",
    "valle.data.hebrew_root_tokenizer",
    "valle.models",
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

if isinstance(sys.modules.get("torchaudio"), MagicMock):
    ta_mock = sys.modules["torchaudio"]
    ta_mock.save.side_effect = lambda filepath, *args, **kwargs: Path(filepath).touch()

if isinstance(sys.modules.get("soundfile"), MagicMock):
    import numpy as np
    sf_mock = sys.modules["soundfile"]
    _mock_audio_store = {}

    def _mock_sf_write(filepath, data=None, sr=24000, **kwargs):
        p = Path(filepath).resolve()
        p.touch()
        if data is not None:
            _mock_audio_store[p] = (np.array(data, dtype=np.float32), sr)

    def _mock_sf_read(filepath, *args, **kwargs):
        p = Path(filepath).resolve()
        if p in _mock_audio_store:
            arr, sr = _mock_audio_store[p]
            return arr.copy(), sr
        return np.ones(24000, dtype=np.float32) * 0.1, 24000

    sf_mock.write.side_effect = _mock_sf_write
    sf_mock.read.side_effect = _mock_sf_read
