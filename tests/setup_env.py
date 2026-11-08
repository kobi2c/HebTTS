import logging
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

# Silence xformers C++/CUDA incompatibility warning (HebTTS does not use xformers C++ ops)
logging.getLogger("xformers").setLevel(logging.ERROR)

# Silence torchvision image extension warning since TTS does not use vision IO
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Silence deprecated weight_norm and custom_fwd/bwd FutureWarnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*weight_norm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*xformers.*")

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
