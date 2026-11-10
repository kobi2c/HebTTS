import base64
from enum import IntEnum
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SpeakerMode(IntEnum):
    """Supported speaker selection modes for synthesis."""
    Predefined = 0
    Clone = 1

    @classmethod
    def _missing_(cls, value: Any) -> "SpeakerMode":
        if isinstance(value, str):
            val_clean = value.strip().lower()
            if val_clean in ("0", "predefined", "predefined speaker", "predefined_speaker", "speaker"):
                return cls.Predefined
            if val_clean in ("1", "clone", "custom voice clone", "custom_voice_clone", "voice clone"):
                return cls.Clone
        elif isinstance(value, int):
            for member in cls:
                if member.value == value:
                    return member
        return super()._missing_(value)

    def __str__(self) -> str:
        return "Predefined Speaker" if self == SpeakerMode.Predefined else "Custom Voice Clone"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            try:
                return self.value == SpeakerMode(other).value
            except (ValueError, TypeError):
                return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class Speaker:
    """Represents a reference speaker voice (predefined or clone)."""
    id       : str = ""
    filename : str = ""
    ext      : str = "wav"
    text     : str = ""
    data     : str = ""
    base     : str = "speakers"
    mode     : SpeakerMode = SpeakerMode.Predefined

    def __post_init__(self) -> None:
        self.mode = SpeakerMode(self.mode)
        if self.ext:
            self.ext = self.ext.lstrip(".").lower()
        elif self.filename:
            self.ext = Path(self.filename).suffix.lstrip(".").lower() or "wav"
        else:
            self.ext = "wav"

    @property
    def full_path(self) -> Path:
        """Returns the complete filesystem path relative to the workspace root or base directory."""
        p = Path(self.filename)
        if p.is_absolute():
            return p
        if self.base:
            return Path(self.base) / p
        return p

    @property
    def exists(self) -> bool:
        """Returns True if the speaker audio file exists on disk."""
        return os.path.exists(self.full_path)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Speaker):
            return (
                self.mode == other.mode
                and self.id == other.id
                and self.filename == other.filename
                and self.text == other.text
                and self.data == other.data
            )
        return False

    def resolve_audio_path(self) -> str:
        """
        Resolves an accessible audio filepath on disk.
        If the file exists locally, returns its path.
        Otherwise, if self.data has base64 audio, decodes to a temporary audio file and returns its path.
        """
        if self.filename and self.exists:
            return str(self.full_path.resolve())
        if self.data:
            ext_suffix = f".{self.ext}" if self.ext else ".wav"
            decoded_bytes = base64.b64decode(self.data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext_suffix) as f:
                f.write(decoded_bytes)
                return f.name
        if self.filename:
            return str(self.full_path)
        raise ValueError(f"Speaker '{self.id}' has neither existing file on disk nor base64 audio data.")

    def load_audio(self, filepath: Union[str, Path]) -> "Speaker":
        """Loads audio file from disk, sets filename/ext/base, and encodes data to base64."""
        p = Path(filepath).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Audio file '{filepath}' not found.")
        self.filename = p.name
        self.ext = p.suffix.lstrip(".").lower() or "wav"
        self.base = str(p.parent)
        with open(p, "rb") as f:
            self.data = base64.b64encode(f.read()).decode("utf-8")
        return self

    def to_dict(self, embed_data: bool = True) -> Dict[str, Any]:
        """Converts speaker to dictionary format. If embed_data is True, embeds base64 audio data."""
        data_str = self.data
        if embed_data and not data_str and self.filename and self.exists:
            try:
                with open(self.full_path, "rb") as f:
                    data_str = base64.b64encode(f.read()).decode("utf-8")
                self.data = data_str
            except Exception:
                pass
        return {
            "id": self.id,
            "filename": self.filename,
            "ext": self.ext,
            "text": self.text,
            "data": data_str,
            "mode": int(self.mode),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base: str = "speakers") -> "Speaker":
        """Constructs a Speaker instance from canonical dictionary representation."""
        spk_id = str(data.get("id", ""))
        filename = str(data.get("filename", ""))
        ext = str(data.get("ext", ""))
        if not ext and filename:
            ext = Path(filename).suffix.lstrip(".").lower() or "wav"
        text = str(data.get("text", ""))
        b64_data = str(data.get("data", ""))
        mode = SpeakerMode(data.get("mode", SpeakerMode.Predefined))
        return cls(
            id=spk_id,
            filename=filename,
            ext=ext or "wav",
            text=text,
            data=b64_data,
            base=base,
            mode=mode,
        )

    @classmethod
    def create(
        cls,
        mode: Union[SpeakerMode, int, str] = SpeakerMode.Predefined,
        id: Optional[str] = None,
        clone_path: Optional[Path] = None,
        clone_text: Optional[str] = None,
        speaker: Optional["Speaker"] = None,
        source: Optional[Dict[str, Any]] = None,
        load_audio: bool = True,
    ) -> "Speaker":
        """
        Factory method to create a Speaker instance with explicit parameters.
        - If speaker is provided, updates and returns it.
        - If source (dict) is provided, deserializes from dictionary.
        - If mode is Predefined, resolves speaker from SpeakerStore using id.
        - If mode is Clone, builds clone from clone_path and clone_text.
        """
        mode = SpeakerMode(mode)
        text_clean = str(clone_text or "").strip()

        if speaker is not None:
            speaker.mode = mode
            if id is not None:
                speaker.id = str(id)
            if clone_text is not None:
                speaker.text = text_clean
            return speaker

        if source is not None:
            spk = cls.from_dict(source)
            spk.mode = mode
            if id is not None:
                spk.id = str(id)
            if clone_text is not None:
                spk.text = text_clean
            return spk

        if mode == SpeakerMode.Predefined:
            spk = SpeakerStore.resolve(id)
            if clone_text is not None:
                spk.text = text_clean
            return spk

        # Clone Mode
        if clone_path is not None:
            ext = clone_path.suffix.lstrip(".").lower() or "wav"
            spk_id = str(id) if id is not None else clone_path.stem
            spk = cls(
                id=spk_id,
                filename=clone_path.name,
                ext=ext,
                text=text_clean,
                base=str(clone_path.parent),
                mode=SpeakerMode.Clone,
            )
            if load_audio and spk.exists:
                spk.load_audio(clone_path)
            return spk

        return cls(
            id=str(id) if id is not None else "clone",
            text=text_clean,
            mode=SpeakerMode.Clone,
        )

    def parse(self, data: dict, base: Union[Path, str], id: str) -> "Speaker":
        """Updates speaker attributes from a dictionary (e.g. speakers.yaml)."""
        base_path = Path(base)
        self.base = base_path.as_posix()

        if id is not None:
            self.id = str(id)

        raw_wav = data.get("filename", data.get("audio-file", data.get("audio-prompt", self.filename or self.id)))
        if raw_wav:
            raw = Path(raw_wav)
            try:
                if raw.is_relative_to(base_path):
                    self.filename = raw.relative_to(base_path).as_posix()
                elif raw.resolve().is_relative_to(base_path.resolve()):
                    self.filename = raw.resolve().relative_to(base_path.resolve()).as_posix()
                else:
                    self.filename = raw.as_posix()
            except Exception:
                self.filename = raw.as_posix()
            self.ext = Path(self.filename).suffix.lstrip(".").lower() or "wav"

        raw_text = data.get("text", data.get("audio-text", data.get("text-prompt", None)))
        if raw_text is not None:
            self.text = str(raw_text)

        if "data" in data and data["data"]:
            self.data = str(data["data"])

        return self


class _StoreClassOrInstanceMethod:
    """Descriptor enabling store methods to function both as Store.method(...) and instance.method(...)."""
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, instance, owner):
        if instance is None:
            return lambda *args, **kwargs: self.fn(owner(), *args, **kwargs)
        return lambda *args, **kwargs: self.fn(instance, *args, **kwargs)


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
            if not sp.id or not sp.filename:
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

        # 2. Discover additional local *.wav and *.mp3 files with companion *.txt files in base directory
        if base_file.exists() and base_file.is_dir():
            # Scan both .wav and .mp3, prioritizing .wav on collision
            discovered_files: Dict[str, Path] = {}
            for f in sorted(base_file.glob("*.mp3")):
                discovered_files[f.stem] = f
            for f in sorted(base_file.glob("*.wav")):
                discovered_files[f.stem] = f  # .wav overwrites .mp3 if both exist

            for stem, f in sorted(discovered_files.items()):
                txt_file = f.with_suffix(".txt")
                if txt_file.exists():
                    try:
                        transcript = txt_file.read_text(encoding="utf-8").strip()
                    except Exception:
                        transcript = ""
                    if transcript:
                        sp = Speaker(
                            id=f.stem,
                            filename=f.relative_to(base_file).as_posix(),
                            ext=f.suffix.lstrip(".").lower() or "wav",
                            text=transcript,
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

    @_StoreClassOrInstanceMethod
    def default(self) -> Speaker:
        """Returns the default speaker (first available from list or fallback)."""
        speakers = self.list()
        return speakers[0] if speakers else Speaker(id="osim", filename="osim.wav", ext="wav", base="speakers", mode=SpeakerMode.Predefined)

    @_StoreClassOrInstanceMethod
    def resolve(self, value: Union[Speaker, Dict[str, Any], str, int, None]) -> Speaker:
        """Resolves a Speaker instance from a Speaker, dict, ID/index string or int, or returns default."""
        if value is None:
            return self.default()
        if isinstance(value, Speaker):
            return value
        if isinstance(value, dict):
            return Speaker.from_dict(value)
        if isinstance(value, (str, int)):
            found = self.find(value)
            if found:
                return found
            return Speaker(id=str(value), mode=SpeakerMode.Predefined)
        return self.default()

    @_StoreClassOrInstanceMethod
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

        # Lookup by ID, audio path, or filename/stem
        target = str_val
        target_name = Path(str_val).name
        target_stem = Path(str_val).stem

        for spk in speakers:
            candidates = {
                spk.id,
                spk.filename,
                Path(spk.filename).name,
                Path(spk.filename).stem,
                spk.full_path.as_posix(),
                spk.full_path.name,
                spk.full_path.stem,
            }
            if (
                target in candidates
                or target_name in candidates
                or target_stem in candidates
            ):
                return spk

        return None
