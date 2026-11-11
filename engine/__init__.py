from typing import Any, Union, Optional
from .base import VoiceEngine
from .heb_tts import HebTTSEngine


def get_engine(
    engine: Optional[Union[VoiceEngine, str]] = None,
    **kwargs: Any,
) -> VoiceEngine:
    """
    Factory helper to resolve a VoiceEngine instance.
    If engine is already a VoiceEngine, returns it.
    If engine is a string ('hebtts', 'heb_tts'), returns HebTTSEngine.
    If engine is None, defaults to HebTTSEngine.
    """
    if isinstance(engine, VoiceEngine):
        return engine
    if engine is None:
        return HebTTSEngine(**kwargs)
    if isinstance(engine, str):
        key = engine.strip().lower().replace("-", "_")
        if key in ("hebtts", "heb_tts", "valle", "valle_hebrew"):
            return HebTTSEngine(**kwargs)
        raise ValueError(f"Unknown VoiceEngine name: '{engine}'")
    raise TypeError(f"Expected VoiceEngine or str, got {type(engine)}")


__all__ = [
    "VoiceEngine",
    "HebTTSEngine",
    "get_engine",
]

