from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EngineInfo:
    """Strongly-typed metadata and runtime status for a VoiceEngine."""
    id               : str
    name             : str
    description      : str = ""
    sample_rate      : int = 24000
    device           : str = "cpu"
    device_name      : str = "CPU"
    is_loaded        : bool = False
    loaded_checkpoint: Optional[str] = None
    languages        : List[str] = field(default_factory=lambda: ["he"])
    last_tokens_str  : Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts EngineInfo instance to a plain dictionary for serialization."""
        return asdict(self)
