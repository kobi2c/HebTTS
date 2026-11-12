import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from .Speaker import Speaker, SpeakerStore, SpeakerMode
from .Checkpoint import Checkpoint, CheckpointStore


class MbdMode(IntEnum):
    Disabled = 0
    Enabled = 1


class SeedMode(IntEnum):
    Disabled = 0
    Enabled = 1

class NikudMode(IntEnum):
    Disabled = 0
    Enabled = 1


class ChainMode(IntEnum):
    Disabled = 0
    Enabled = 1


DEFAULT_TEXT: str = (
    "שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית.\n"
    "המודל מאפשר שכפול קול ויצירת דיבור טבעי באיכות גבוהה."
)


class _LoadJsonDescriptor:
    """Descriptor enabling loadJson to function both as Config.loadJson(data) and instance.loadJson(data)."""
    def __get__(self, instance, owner):
        if instance is None:
            def _class_load(json_data: Union[str, Path, Dict[str, Any]]) -> "Config":
                return owner().load(json_data)
            return _class_load
        else:
            def _instance_load(json_data: Union[str, Path, Dict[str, Any]]) -> "Config":
                return instance.load(json_data)
            return _instance_load


class _FromDictDescriptor:
    """Descriptor enabling from_dict to function both as Config.from_dict(d) and instance.from_dict(d)."""
    def __get__(self, instance, owner):
        if instance is None:
            def _class_from_dict(data: Dict[str, Any]) -> "Config":
                return owner()._apply_dict(data)
            return _class_from_dict
        else:
            def _instance_from_dict(data: Dict[str, Any]) -> "Config":
                return instance._apply_dict(data)
            return _instance_from_dict


@dataclass
class Config:
    checkpoint         : str = "checkpoint.pt"
    speaker            : Speaker = field(default_factory=SpeakerStore.default)
    text               : str = DEFAULT_TEXT
    pause_duration     : float = 0.3
    temperature        : float = 1.0
    top_k              : int = 50
    mbd                : MbdMode = MbdMode.Disabled
    seed               : int = 420
    seed_mode          : SeedMode = SeedMode.Enabled
    nikud_mode         : NikudMode = NikudMode.Enabled
    chain_speaker      : ChainMode = ChainMode.Disabled
    max_chunk_size     : int = 100
    fade_ms            : float = 20.0
    trim_noise         : bool = True
    pad_ms             : float = 30.0

    def __post_init__(self) -> None:
        self.speaker = SpeakerStore.resolve(self.speaker)

    def reset(self) -> "Config":
        """Resets all configuration fields in-place to their default values."""
        checkpoints = [c.path for c in CheckpointStore().list()]
        default_checkpoint = checkpoints[0] if checkpoints else "checkpoint.pt"

        self.checkpoint = default_checkpoint
        self.speaker = SpeakerStore.default()
        self.text = DEFAULT_TEXT
        self.pause_duration = 0.3
        self.temperature = 1.0
        self.top_k = 50
        self.mbd = MbdMode.Disabled
        self.seed = 420
        self.seed_mode = SeedMode.Enabled
        self.nikud_mode = NikudMode.Enabled
        self.chain_speaker = ChainMode.Disabled
        self.max_chunk_size = 100
        self.fade_ms = 20.0
        self.trim_noise = True
        self.pad_ms = 30.0
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration fields to a standard dictionary."""
        return {
            "checkpoint": self.checkpoint,
            "speaker": self.speaker.to_dict(embed_data=True) if self.speaker else {},
            "text": self.text,
            "pause_duration": float(self.pause_duration),
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "mbd": int(self.mbd),
            "seed": int(self.seed),
            "seed_mode": int(self.seed_mode),
            "nikud_mode": int(self.nikud_mode),
            "chain_speaker": int(self.chain_speaker),
            "max_chunk_size": int(self.max_chunk_size),
            "fade_ms": float(self.fade_ms),
            "trim_noise": bool(self.trim_noise),
            "pad_ms": float(self.pad_ms),
        }

    def _apply_dict(self, data: Dict[str, Any]) -> "Config":
        """Updates configuration fields in-place from a dictionary."""
        if "checkpoint" in data and data["checkpoint"] is not None:
            self.checkpoint = str(data["checkpoint"])
        if "speaker" in data and data["speaker"] is not None:
            self.speaker = SpeakerStore.resolve(data["speaker"])
        if "text" in data and data["text"] is not None:
            self.text = str(data["text"])
        if "pause_duration" in data and data["pause_duration"] is not None:
            self.pause_duration = float(data["pause_duration"])
        if "temperature" in data and data["temperature"] is not None:
            self.temperature = float(data["temperature"])
        if "top_k" in data and data["top_k"] is not None:
            self.top_k = int(data["top_k"])
        if "mbd" in data and data["mbd"] is not None:
            self.mbd = MbdMode(int(data["mbd"]))
        if "seed" in data and data["seed"] is not None:
            self.seed = int(data["seed"])
        if "seed_mode" in data and data["seed_mode"] is not None:
            self.seed_mode = SeedMode(int(data["seed_mode"]))
        if "nikud_mode" in data and data["nikud_mode"] is not None:
            self.nikud_mode = NikudMode(int(data["nikud_mode"]))
        if "chain_speaker" in data and data["chain_speaker"] is not None:
            self.chain_speaker = ChainMode(int(data["chain_speaker"]))
        if "max_chunk_size" in data and data["max_chunk_size"] is not None:
            self.max_chunk_size = int(data["max_chunk_size"])
        if "fade_ms" in data and data["fade_ms"] is not None:
            self.fade_ms = float(data["fade_ms"])
        if "trim_noise" in data and data["trim_noise"] is not None:
            self.trim_noise = bool(data["trim_noise"])
        if "pad_ms" in data and data["pad_ms"] is not None:
            self.pad_ms = float(data["pad_ms"])
        return self

    from_dict = _FromDictDescriptor()

    def dumpJson(self, indent: int = 4) -> str:
        """Serializes configuration to a standalone JSON string with embedded base64 audio."""
        d = self.to_dict()
        return json.dumps(d, indent=indent, ensure_ascii=False)

    def saveJson(self, file_path: Union[str, Path], indent: int = 4) -> str:
        """Saves configuration JSON string to a file on disk."""
        p = Path(file_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.dumpJson(indent=indent)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return str(p)

    def load(self, json_data: Union[str, Path, Dict[str, Any]]) -> "Config":
        """Loads and updates configuration in-place from a JSON string, file path, or dictionary."""
        parsed_dict: Dict[str, Any] = {}
        if isinstance(json_data, dict):
            parsed_dict = json_data
        elif isinstance(json_data, (str, Path)):
            p = Path(json_data)
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    parsed_dict = json.load(f)
            else:
                parsed_dict = json.loads(str(json_data))
        else:
            raise TypeError(f"Unsupported json_data type: {type(json_data)}")
        return self.from_dict(parsed_dict)

    loadJson = _LoadJsonDescriptor()
