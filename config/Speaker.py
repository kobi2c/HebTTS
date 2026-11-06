import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Speaker:
    """Represents a predefined reference speaker voice."""
    id   : str = ""
    txt  : str = ""
    wav  : str = ""
    base : str = "speakers"

    @property
    def full_path(self) -> Path:
        """Returns the complete filesystem path relative to the workspace root."""
        p = Path(self.wav)
        if p.is_absolute():
            return p
        if self.base:
            return Path(self.base) / p
        else:
            return p

    @property
    def exists(self) -> bool:
        """Returns True if the speaker audio file exists on disk."""
        return os.path.exists(self.full_path)

    def parse(self, data: dict, base: Union[Path, str], id: str) -> "Speaker":
        """Updates speaker attributes from a dictionary."""
        base_path = Path(base)
        self.base = base_path.as_posix()

        if id is not None:
            self.id = str(id)

        raw_wav = data.get("audio-file", data.get("audio-prompt", self.wav or self.id))
        if raw_wav:
            raw = Path(raw_wav)
            try:
                if raw.is_relative_to(base_path):
                    self.wav = raw.relative_to(base_path).as_posix()
                elif raw.resolve().is_relative_to(base_path.resolve()):
                    self.wav = raw.resolve().relative_to(base_path.resolve()).as_posix()
                else:
                    self.wav = raw.as_posix()
            except Exception:
                self.wav = raw.as_posix()

        raw_text = data.get("audio-text", data.get("text-prompt", None))
        if raw_text is not None:
            self.txt = str(raw_text)

        return self


class SpeakerStore:
    """Manages loading, caching, and querying of predefined speakers from speakers/speakers.yaml."""

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        self._main_path: str = self.find_path(path)
        self._load_path: str = self._main_path
        self._list: Optional[List[Speaker]] = None

    @classmethod
    def find_path(cls, path: Union[str, Path, None] = None) -> str:
        """Finds the path to the speakers configuration file."""
        if path is not None:
            return str(path)
        return "speakers/speakers.yaml"

    def load(self, path: Union[str, Path, None] = None) -> List[Speaker]:
        """Loads speakers from YAML and discovers additional local *.wav files with companion *.txt transcripts."""
        self._load_path = str(path) if path is not None else self._main_path
        yaml_file = Path(self._load_path)
        base_file = yaml_file.parent

        raw_dict: Dict[str, Any] = {}
        if yaml_file.exists():
            try:
                import yaml
                with open(yaml_file, "r", encoding="utf-8") as f:
                    raw_dict = yaml.safe_load(f) or {}
            except ImportError:
                try:
                    from omegaconf import OmegaConf
                    loaded = OmegaConf.load(str(yaml_file))
                    raw_dict = OmegaConf.to_container(loaded, resolve=True) or {}
                except Exception:
                    raw_dict = {}

        speakers: List[Speaker] = []
        seen_paths: set = set()

        def add(sp: Speaker, check_path: bool) -> None:
            if not sp.id or not sp.wav:
                return
            cpath = sp.full_path.resolve().as_posix().lower()
            if check_path and cpath in seen_paths:
                return

            speakers.append(sp)
            seen_paths.add(cpath)

        # 1. Load speakers from YAML
        for spk_id, spk_data in raw_dict.items():
            if not isinstance(spk_data, dict):
                continue
            sp = Speaker().parse(spk_data, base_file, spk_id)
            add(sp, False)

        # 2. Discover additional local *.wav files with companion *.txt files in base directory
        if base_file.exists() and base_file.is_dir():
            for f in sorted(base_file.glob("*.wav")):
                txt_file = f.with_suffix(".txt")
                if txt_file.exists():
                    try:
                        transcript = txt_file.read_text(encoding="utf-8").strip()
                    except Exception:
                        transcript = ""
                    if transcript:
                        sp = Speaker(
                            id=f.stem,
                            txt=transcript,
                            wav=f.relative_to(base_file).as_posix(),
                            base=base_file.as_posix(),
                        )
                        add(sp, True)

        self._list = speakers
        return self._list

    def list(self) -> List[Speaker]:
        """Returns the list of loaded Speaker instances (loads lazily if not yet loaded)."""
        if self._list is None:
            self.load()
        return self._list or []

    def clear(self) -> None:
        """Clears the cached speakers list."""
        self._list = None

    def find(self, index_or_id: Union[int, str]) -> Optional[Speaker]:
        """
        Finds a speaker by 1-based index (int or digit string) or by case-insensitive speaker ID.
        Returns matching Speaker instance or None if not found.
        """
        speakers = self.list()
        if not speakers:
            return None

        # Check for 1-based integer index or digit string within range
        str_val = str(index_or_id).strip()
        if str_val.isdigit():
            idx = int(str_val)
            if 1 <= idx <= len(speakers):
                return speakers[idx - 1]
            else:
                return None

        # Lookup by speaker ID (case-insensitive)
        target = str_val.lower()
        for spk in speakers:
            if spk.id.lower() == target:
                return spk

        return None


# Backward-compatible alias
SpeakerData = SpeakerStore
