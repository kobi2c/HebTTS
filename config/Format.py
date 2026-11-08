from enum import IntEnum
from typing import Any


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
