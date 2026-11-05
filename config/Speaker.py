import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Speaker:
    """Represents a predefined reference speaker voice."""
    id: str
    txt: str
    wav: str
    wav_base: str


class SpeakerManager:
    """Manages loading, caching, and querying of predefined speakers."""

    def __init__(self, default_yaml: Union[str, Path] = "speakers/speakers.yaml") -> None:
        self.default_yaml: str = str(default_yaml)
        self._list: Optional[List[Speaker]] = None

    def load(self, yaml_path: Optional[Union[str, Path]] = None) -> List[Speaker]:
        """Loads speakers from a YAML file, builds Speaker instances, and caches them."""
        path = Path(yaml_path or self.default_yaml).resolve()
        if not path.exists():
            self._list = []
            return self._list

        raw_dict: Dict[str, Any] = {}
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                raw_dict = yaml.safe_load(f) or {}
        except ImportError:
            try:
                from omegaconf import OmegaConf
                loaded = OmegaConf.load(str(path))
                raw_dict = OmegaConf.to_container(loaded, resolve=True) or {}
            except Exception:
                raw_dict = {}

        speakers: List[Speaker] = []
        base_dir = path.parent

        for spk_id, spk_data in raw_dict.items():
            if not isinstance(spk_data, dict):
                continue
            txt = str(spk_data.get("text-prompt", ""))
            raw_wav = str(spk_data.get("audio-prompt", ""))
            resolved_wav = str((base_dir / raw_wav).resolve())
            speakers.append(
                Speaker(
                    id=str(spk_id),
                    txt=txt,
                    wav=resolved_wav,
                    wav_base=raw_wav,
                )
            )

        self._list = speakers
        return self._list

    def list(self) -> List[Speaker]:
        """Returns the list of loaded Speaker instances (loads lazily if not yet loaded)."""
        if self._list is None:
            self.load()
        return self._list or []

    def find(self, index_or_id: Union[int, str]) -> Optional[Speaker]:
        """
        Finds a speaker by 1-based index (int or digit string) or by case-insensitive speaker ID.
        Returns matching Speaker instance or None if not found.
        """
        speakers = self.list()
        if not speakers:
            return None

        # Check for 1-based integer index or digit string
        if isinstance(index_or_id, int):
            idx = index_or_id
            if 1 <= idx <= len(speakers):
                return speakers[idx - 1]
            return None

        str_val = str(index_or_id).strip()
        if str_val.isdigit():
            idx = int(str_val)
            if 1 <= idx <= len(speakers):
                return speakers[idx - 1]
            return None

        # Lookup by speaker ID (case-insensitive)
        target = str_val.lower()
        for spk in speakers:
            if spk.id.lower() == target:
                return spk

        return None

    def clear(self) -> None:
        """Clears the cached speakers list."""
        self._list = None


# Singleton instance for global access
Speakers: SpeakerManager = SpeakerManager()

