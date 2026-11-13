import sys
from .HebTTSEngine import HebTTSEngine

# Backward-compatibility alias for legacy imports
sys.modules.setdefault("engine.heb_tts.engine", sys.modules[__name__])

__all__ = ["HebTTSEngine"]

