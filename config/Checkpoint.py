import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union


@dataclass
class Checkpoint:
    """Represents a neural model checkpoint file."""
    id: str
    path: str
    description: str = ""
    gdrive_id: Optional[str] = None

    @property
    def exists(self) -> bool:
        """Returns True if the checkpoint file exists on disk."""
        return os.path.exists(self.path)


class CheckpointManager:
    """Manages discovery, caching, and querying of model checkpoints."""

    KNOWN_CHECKPOINTS: ClassVar[Dict[int, Dict[str, Any]]] = {
        1: {
            "id": "checkpoint.pt",
            "description": "VALL-E Hebrew (AR 1.2M / NAR 200K steps on HebDB)",
            "path": "checkpoint.pt",
            "gdrive_id": "11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv",
        }
    }

    def __init__(self, search_dir: Union[str, Path] = ".") -> None:
        self.search_dir: str = str(search_dir)
        self._list: Optional[List[Checkpoint]] = None

    def load(self, search_dir: Optional[Union[str, Path]] = None) -> List[Checkpoint]:
        """Loads known checkpoints and discovers local *.pt files in search_dir."""
        target_dir = Path(search_dir or self.search_dir).resolve()
        checkpoints: List[Checkpoint] = []
        seen_paths: set = set()

        # 1. Add known official checkpoints
        for num, meta in sorted(self.KNOWN_CHECKPOINTS.items(), key=lambda x: x[0]):
            rel_path = meta["path"]
            resolved_path = str((target_dir / rel_path).resolve()) if not Path(rel_path).is_absolute() else rel_path
            # Store standard relative path if inside target_dir or matches
            stored_path = rel_path if os.path.exists(rel_path) or not os.path.exists(resolved_path) else resolved_path
            checkpoints.append(
                Checkpoint(
                    id=meta["id"],
                    path=stored_path,
                    description=meta.get("description", ""),
                    gdrive_id=meta.get("gdrive_id"),
                )
            )
            seen_paths.add(stored_path)
            seen_paths.add(resolved_path)
            seen_paths.add(meta["id"])

        # 2. Discover additional local *.pt files
        if target_dir.exists() and target_dir.is_dir():
            for f in sorted(target_dir.glob("*.pt")):
                f_str = f.name
                f_full = str(f)
                if f_str not in seen_paths and f_full not in seen_paths:
                    checkpoints.append(
                        Checkpoint(
                            id=f_str,
                            path=f_str,
                            description=f"Local checkpoint ({f_str})",
                        )
                    )
                    seen_paths.add(f_str)
                    seen_paths.add(f_full)

        self._list = checkpoints
        return self._list

    def list(self) -> List[Checkpoint]:
        """Returns the list of loaded Checkpoint instances (loads lazily if not yet loaded)."""
        if self._list is None:
            self.load()
        return self._list or []

    def find(self, index_or_id: Union[int, str]) -> Optional[Checkpoint]:
        """
        Finds a checkpoint by 1-based index (int or digit string), by ID/filename,
        or as a direct file path on disk. Returns Checkpoint or None if not found.
        """
        checkpoints = self.list()

        # Check for 1-based integer index or digit string
        if isinstance(index_or_id, int):
            idx = index_or_id
            if 1 <= idx <= len(checkpoints):
                return checkpoints[idx - 1]
            return None

        str_val = str(index_or_id).strip()
        if str_val.isdigit():
            idx = int(str_val)
            if 1 <= idx <= len(checkpoints):
                return checkpoints[idx - 1]
            return None

        # Lookup by ID or path (case-insensitive)
        target = str_val.lower()
        for cp in checkpoints:
            if cp.id.lower() == target or cp.path.lower() == target or Path(cp.path).name.lower() == target:
                return cp

        # Direct file check if file exists on disk
        p = Path(str_val)
        if p.exists() and p.is_file():
            return Checkpoint(id=p.name, path=str_val, description=f"Local checkpoint ({p.name})")

        return None

    def clear(self) -> None:
        """Clears the cached checkpoints list."""
        self._list = None


# Singleton instance for global access
Checkpoints: CheckpointManager = CheckpointManager()

