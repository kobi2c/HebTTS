from enum import IntEnum
from typing import Any, List, Union


class Format(IntEnum):
    """Supported audio output formats for synthesis."""
    WAV = 1
    MP3 = 2

    @classmethod
    def _missing_(cls, value: Any) -> "Format":
        if isinstance(value, str):
            val_lower = value.strip().lower()
            for member in cls:
                if member.name.lower() == val_lower:
                    return member
        elif isinstance(value, int):
            for member in cls:
                if member.value == value:
                    return member
        return super()._missing_(value)

    def __str__(self) -> str:
        return self.name.lower()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.name.lower() == other.strip().lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    @property
    def extension(self) -> str:
        """Returns the file extension with dot (e.g. '.wav', '.mp3')."""
        return f".{self.name.lower()}"

    @classmethod
    def extensions(cls) -> List[str]:
        """Returns list of supported audio file extensions (e.g. ['.wav', '.mp3'])."""
        return [f".{m.name.lower()}" for m in cls]

    @classmethod
    def names(cls) -> List[str]:
        """Returns list of supported format names in lowercase (e.g. ['wav', 'mp3'])."""
        return [m.name.lower() for m in cls]

    @classmethod
    def normalize(cls, val: Any) -> List["Format"]:
        """Normalizes format inputs (single, string, list, or 'both') into a List[Format]."""
        if val is None:
            return [cls.WAV]
        if isinstance(val, (Format, str, int)):
            val = [val]
        formats: List["Format"] = []
        for item in val:
            if isinstance(item, cls):
                if item not in formats:
                    formats.append(item)
            elif isinstance(item, str):
                s = item.strip().lower()
                if s == "both":
                    for f in (cls.WAV, cls.MP3):
                        if f not in formats:
                            formats.append(f)
                else:
                    f = cls(s)
                    if f not in formats:
                        formats.append(f)
            elif isinstance(item, int):
                f = cls(item)
                if f not in formats:
                    formats.append(f)
        return formats or [cls.WAV]

    normalize_formats = normalize

