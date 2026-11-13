import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union
from .EngineInfo import EngineInfo
from .VoiceEngine import VoiceEngine
from .heb_tts import HebTTSEngine


def _zonos_loader(**kwargs: Any) -> VoiceEngine:
    try:
        from .zonos import ZonosEngine
    except ImportError as e:
        raise ImportError(
            f"ZonosEngine dependencies could not be imported: {e}. "
            "Ensure Zonos packages (transformers, inflect, kanjize, etc.) are installed."
        ) from e
    return ZonosEngine(**kwargs)


_ENGINE_REGISTRY: Dict[str, Union[Type[VoiceEngine], Callable[..., VoiceEngine]]] = {}
_ENGINE_ALIASES: Dict[str, str] = {}


def register_engine(
    name_or_aliases: Union[str, List[str]],
    engine_cls_or_factory: Union[Type[VoiceEngine], Callable[..., VoiceEngine]],
) -> None:
    """Registers a VoiceEngine class or loader under one or more identifier names."""
    if isinstance(name_or_aliases, str):
        names = [name_or_aliases]
    else:
        names = list(name_or_aliases)

    if not names:
        raise ValueError("At least one engine name or alias must be provided.")

    canonical_id = names[0].strip().lower().replace("-", "_")
    _ENGINE_REGISTRY[canonical_id] = engine_cls_or_factory

    for alias in names:
        norm_alias = alias.strip().lower().replace("-", "_")
        _ENGINE_ALIASES[norm_alias] = canonical_id


def unregister_engine(engine_id: str) -> None:
    """Removes a registered engine and its aliases."""
    norm_id = engine_id.strip().lower().replace("-", "_")
    canonical_id = _ENGINE_ALIASES.get(norm_id, norm_id)
    _ENGINE_REGISTRY.pop(canonical_id, None)
    aliases_to_remove = [k for k, v in _ENGINE_ALIASES.items() if v == canonical_id or k == norm_id]
    for k in aliases_to_remove:
        _ENGINE_ALIASES.pop(k, None)


def list_engines() -> List[str]:
    """Returns a list of all canonical registered engine identifiers."""
    return sorted(list(_ENGINE_REGISTRY.keys()))


def get_engine_class(engine_id: str) -> Type[VoiceEngine]:
    """Resolves and returns the underlying VoiceEngine class for an engine identifier."""
    norm_id = engine_id.strip().lower().replace("-", "_")
    canonical = _ENGINE_ALIASES.get(norm_id, norm_id)
    if canonical not in _ENGINE_REGISTRY:
        raise ValueError(f"Unknown VoiceEngine name: '{engine_id}'")

    target = _ENGINE_REGISTRY[canonical]
    if isinstance(target, type) and issubclass(target, VoiceEngine):
        return target
    # If registered as a loader (e.g. Zonos), resolve dynamically
    if canonical == "zonos":
        from .zonos import ZonosEngine
        return ZonosEngine
    # If callable, call without arguments to obtain class
    inst = target()
    return type(inst)


def get_engine_info(engine_id: str) -> EngineInfo:
    """Retrieves EngineInfo metadata for an engine without fully instantiating it if possible."""
    return get_engine_class(engine_id).info


def get_engine(
    engine: Optional[Union[VoiceEngine, str]] = None,
    **kwargs: Any,
) -> VoiceEngine:
    """
    Factory helper to resolve a VoiceEngine instance.
    If engine is already a VoiceEngine, returns it.
    If engine is a string, resolves it from the registry.
    If engine is None, defaults to HebTTSEngine.
    """
    if isinstance(engine, VoiceEngine):
        return engine
    if engine is None:
        return HebTTSEngine(**kwargs)
    if isinstance(engine, str):
        norm_key = engine.strip().lower().replace("-", "_")
        canonical = _ENGINE_ALIASES.get(norm_key, norm_key)
        if canonical in _ENGINE_REGISTRY:
            factory = _ENGINE_REGISTRY[canonical]
            return factory(**kwargs)
        raise ValueError(f"Unknown VoiceEngine name: '{engine}'")
    raise TypeError(f"Expected VoiceEngine or str, got {type(engine)}")


# Pre-register default engines
register_engine(["hebtts", "heb_tts", "valle", "valle_hebrew"], HebTTSEngine)
register_engine(["zonos", "zonos_hebrew", "zonos_engine"], _zonos_loader)


def __getattr__(name: str) -> Any:
    if name == "ZonosEngine":
        from .zonos import ZonosEngine
        return ZonosEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EngineInfo",
    "VoiceEngine",
    "HebTTSEngine",
    "ZonosEngine",
    "get_engine",
    "register_engine",
    "unregister_engine",
    "list_engines",
    "get_engine_class",
    "get_engine_info",
]
