import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from .Speaker import Speaker, SpeakerData
from .Checkpoint import Checkpoint, CheckpointStore


class MbdMode(IntEnum):
    Disabled = 0
    Enabled = 1


class RandomMode(IntEnum):
    Disabled = 0
    Enabled = 1

class NikudMode(IntEnum):
    Disabled = 0
    Enabled = 1


DEFAULT_TEXT: str = (
    "שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית.\n"
    "המודל מאפשר שכפול קול ויצירת דיבור טבעי באיכות גבוהה."
)


def audio_to_base64(filepath: Optional[str]) -> Optional[Dict[str, str]]:
    """Encodes an audio file on disk to a self-contained base64 dictionary."""
    if not filepath or not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(filepath)[1] or ".wav"
    return {"ext": ext, "data": encoded}


def base64_to_audio(b64_dict: Optional[Union[Dict[str, str], str]]) -> Optional[str]:
    """Decodes a base64 dictionary to a temporary audio file on disk."""
    if not b64_dict:
        return None
    if isinstance(b64_dict, str):
        return b64_dict if os.path.exists(b64_dict) else None
    if not isinstance(b64_dict, dict) or "data" not in b64_dict:
        return None
    ext = b64_dict.get("ext", ".wav")
    decoded = base64.b64decode(b64_dict["data"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(decoded)
        return f.name


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
    speaker_mode       : str = "Predefined Speaker"
    predefined_speaker : str = "1"
    clone_audio        : Optional[str] = None
    clone_text         : str = ""
    text               : str = DEFAULT_TEXT
    pause_duration     : float = 0.3
    temperature        : float = 1.0
    top_k              : int = 50
    mbd                : MbdMode = MbdMode.Disabled
    seed_number        : int = 420
    randomize_seed     : RandomMode = RandomMode.Enabled
    nikud_converter    : NikudMode = NikudMode.Enabled

    def reset(self) -> "Config":
        """Resets all configuration fields in-place to their default values."""
        checkpoints = [c.path for c in CheckpointStore().list()]
        default_checkpoint = checkpoints[0] if checkpoints else "checkpoint.pt"
        speakers = SpeakerData().list()
        default_speaker = speakers[0].id if speakers else "osim"

        self.checkpoint = default_checkpoint
        self.speaker_mode = "Predefined Speaker"
        self.predefined_speaker = default_speaker
        self.clone_audio = None
        self.clone_text = ""
        self.text = DEFAULT_TEXT
        self.pause_duration = 0.3
        self.temperature = 1.0
        self.top_k = 50
        self.mbd = MbdMode.Disabled
        self.seed_number = 420
        self.randomize_seed = RandomMode.Enabled
        self.nikud_converter = NikudMode.Enabled
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration fields to a standard dictionary."""
        return {
            "checkpoint": self.checkpoint,
            "speaker_mode": self.speaker_mode,
            "predefined_speaker": self.predefined_speaker,
            "clone_audio": self.clone_audio,
            "clone_text": self.clone_text,
            "text": self.text,
            "pause_duration": float(self.pause_duration),
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "mbd": int(self.mbd),
            "seed_number": int(self.seed_number),
            "randomize_seed": int(self.randomize_seed),
            "nikud_converter": int(self.nikud_converter),
        }

    def _apply_dict(self, data: Dict[str, Any]) -> "Config":
        """Updates configuration fields in-place from a dictionary."""
        if "checkpoint" in data and data["checkpoint"] is not None:
            self.checkpoint = str(data["checkpoint"])
        if "speaker_mode" in data and data["speaker_mode"] is not None:
            self.speaker_mode = str(data["speaker_mode"])
        if "predefined_speaker" in data and data["predefined_speaker"] is not None:
            self.predefined_speaker = str(data["predefined_speaker"])
        if "clone_audio" in data:
            raw_audio = data["clone_audio"]
            if isinstance(raw_audio, dict) and "data" in raw_audio:
                self.clone_audio = base64_to_audio(raw_audio)
            else:
                self.clone_audio = raw_audio
        if "clone_text" in data and data["clone_text"] is not None:
            self.clone_text = str(data["clone_text"])
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
        if "seed_number" in data and data["seed_number"] is not None:
            self.seed_number = int(data["seed_number"])
        if "randomize_seed" in data and data["randomize_seed"] is not None:
            self.randomize_seed = RandomMode(int(data["randomize_seed"]))
        if "nikud_converter" in data and data["nikud_converter"] is not None:
            self.nikud_converter = NikudMode(int(data["nikud_converter"]))
        return self

    from_dict = _FromDictDescriptor()

    def dumpJson(self, indent: int = 4) -> str:
        """Serializes configuration to a JSON string with base64 encoded audio."""
        d = self.to_dict()
        if self.clone_audio and isinstance(self.clone_audio, str) and os.path.exists(self.clone_audio):
            d["clone_audio"] = audio_to_base64(self.clone_audio)
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
