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
    "engine.heb_tts.valle",
    "engine.heb_tts.valle.data",
    "engine.heb_tts.valle.data.collation",
    "engine.heb_tts.valle.data.hebrew_root_tokenizer",
    "engine.heb_tts.valle.models",
    "safetensors",
    "transformers",
    "zonos",
    "zonos.model",
    "zonos.conditioning",
    "zonos.utils",
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

if "gdown" not in sys.modules:
    try:
        __import__("gdown")
    except ImportError:
        mock_gdown = MagicMock()
        def _fake_gdown_download(*args, **kwargs):
            out = kwargs.get("output") or (args[1] if len(args) > 1 else None)
            if out:
                p = Path(out)
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_bytes(b"dummy_weights")
        mock_gdown.download.side_effect = _fake_gdown_download
        sys.modules["gdown"] = mock_gdown

if isinstance(sys.modules.get("torch"), MagicMock):
    import numpy as np
    torch_mock = sys.modules["torch"]

    def _mock_cat(tensors, dim=-1):
        res = MagicMock()
        total_len = sum(
            t.shape[-1] if hasattr(t, "shape") and isinstance(t.shape, (list, tuple)) else 2400
            for t in tensors
        )
        res.shape = (1, total_len)
        res.squeeze.return_value.cpu.return_value.numpy.return_value = np.ones(total_len, dtype=np.float32) * 0.5
        return res

    torch_mock.cat.side_effect = _mock_cat
    torch_mock.zeros.side_effect = lambda shape, *args, **kwargs: MagicMock(
        ndim=len(shape) if isinstance(shape, (tuple, list)) else 1,
        shape=shape,
        squeeze=MagicMock(return_value=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.zeros(shape[-1] if isinstance(shape, (tuple, list)) else 2400, dtype=np.float32))))))
    )

    def _mock_from_numpy(arr):
        res = MagicMock()
        res.ndim = getattr(arr, "ndim", 1)
        res.shape = getattr(arr, "shape", (len(arr),) if hasattr(arr, "__len__") else (24000,))
        res.numpy.return_value = arr
        res.detach.return_value = res
        res.cpu.return_value = res
        res.unsqueeze.side_effect = lambda dim: MagicMock(
            ndim=res.ndim + 1,
            shape=(1, *res.shape),
            numpy=MagicMock(return_value=arr),
            t=MagicMock(return_value=res),
        )
        res.t.side_effect = lambda: res
        return res

    torch_mock.from_numpy.side_effect = _mock_from_numpy

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
