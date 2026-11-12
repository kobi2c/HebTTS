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
        if key in ("zonos", "zonos_hebrew", "zonos_engine"):
            try:
                from .zonos import ZonosEngine
            except ImportError as e:
                raise ImportError(
                    f"ZonosEngine dependencies could not be imported: {e}. "
                    "Ensure Zonos packages (transformers, inflect, kanjize, etc.) are installed."
                ) from e
            return ZonosEngine(**kwargs)
        raise ValueError(f"Unknown VoiceEngine name: '{engine}'")
    raise TypeError(f"Expected VoiceEngine or str, got {type(engine)}")


def __getattr__(name: str) -> Any:
    if name == "ZonosEngine":
        from .zonos import ZonosEngine
        return ZonosEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VoiceEngine",
    "HebTTSEngine",
    "ZonosEngine",
    "get_engine",
]

