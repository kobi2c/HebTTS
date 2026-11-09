import warnings

# Silence torchvision image extension warning since TTS does not use vision IO
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Silence deprecated weight_norm FutureWarnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*weight_norm.*")

try:
    import utils  # Applies torch.load compatibility, audio IO fallbacks, and warning filters
except ImportError:
    pass

from . import data, models, modules, utils

