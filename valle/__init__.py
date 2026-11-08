import logging
import warnings

# Silence xformers C++/CUDA incompatibility warning (HebTTS does not use xformers C++ ops)
logging.getLogger("xformers").setLevel(logging.ERROR)

# Silence torchvision image extension warning since TTS does not use vision IO
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Silence deprecated weight_norm and custom_fwd/bwd FutureWarnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*weight_norm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*xformers.*")

try:
    import utils  # Applies torch.load compatibility, audio IO fallbacks, and warning filters
except ImportError:
    pass

from . import data, models, modules, utils

