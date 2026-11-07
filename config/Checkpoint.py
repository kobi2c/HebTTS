import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Checkpoint:
    """Represents a neural model checkpoint file."""
    id        : str = ""
    path      : str = ""
    info      : str = ""
    gdrive_id : Optional[str] = None
    base      : str = "checkpoints"

    @property
    def full_path(self) -> Path:
        """Returns the complete filesystem path relative to the workspace root."""
        p = Path(self.path)
        if p.is_absolute():
            return p
        if self.base:
            return Path(self.base) / p
        else:
            return p

    @property
    def exists(self) -> bool:
        """Returns True if the checkpoint file exists on disk."""
        return os.path.exists(self.full_path)

    def download(self, quiet: bool = False) -> Path:
        """
        Downloads the checkpoint file from Google Drive if it does not exist on disk.
        Uses the gdown python package with fallback to gdown CLI.
        """
        if self.exists:
            return self.full_path

        if not self.gdrive_id:
            raise ValueError(f"Checkpoint '{self.id}' does not exist on disk and has no gdrive_id.")

        out_path = self.full_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import gdown
            print(f"Downloading checkpoint '{self.id}' ({self.gdrive_id}) via gdown...")
            gdown.download(id=self.gdrive_id, output=str(out_path), quiet=quiet, resume=True)
        except ImportError:
            print(f"Downloading checkpoint '{self.id}' via gdown CLI...")
            try:
                subprocess.run(["gdown", self.gdrive_id, "-O", str(out_path), "--continue"], check=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download checkpoint using gdown CLI: {e}. "
                    f"Please install gdown ('pip install gdown') or run: "
                    f"docker run --rm -v %CD%:/app hebtts:latest gdown {self.gdrive_id} -O {self.full_path}"
                )

        if not out_path.exists():
            raise FileNotFoundError(f"Download completed but '{out_path}' was not found on disk.")

        return out_path


    def parse(self, data: dict, base: Union[Path, str], id: str) -> "Checkpoint":
        """Updates checkpoint attributes from a dictionary."""
        base_path = Path(base)
        self.base = base_path.as_posix()

        if id is not None:
            self.id = str(id)

        path = data.get("path", self.path or self.id)
        if path:
            raw = Path(path)
            try:
                if raw.is_relative_to(base_path):
                    self.path = raw.relative_to(base_path).as_posix()
                elif raw.resolve().is_relative_to(base_path.resolve()):
                    self.path = raw.resolve().relative_to(base_path.resolve()).as_posix()
                else:
                    self.path = raw.as_posix()
            except Exception:
                self.path = raw.as_posix()

        info = data.get("info", data.get("description", self.info))
        if info is not None:
            self.info = str(info)

        gid = data.get("gdrive_id", self.gdrive_id)
        if gid is not None:
            self.gdrive_id = str(gid)

        return self


class CheckpointStore:
    """Manages discovery, caching, and querying of model checkpoints from checkpoints/checkpoints.yaml."""

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        self._main_path: str = self.find_path(path)
        self._load_path: str = self._main_path
        self._list: Optional[List[Checkpoint]] = None

    @classmethod
    def find_path(cls, path: Union[str, Path, None] = None) -> str:
        """Finds the path to the checkpoints configuration file."""
        if path is not None:
            return str(path)
        return "checkpoints/checkpoints.yaml"

    def load(self, path: Union[str, Path, None] = None) -> List[Checkpoint]:
        """Loads checkpoints from YAML and discovers additional *.pt files in the checkpoints directory."""
        self._load_path = str(path) if path is not None else self._main_path
        yaml_file = Path(self._load_path)
        base_file  = yaml_file.parent

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

        checkpoints: List[Checkpoint] = []
        seen_paths: set = set()

        def add(cp: Checkpoint, check_path: bool) -> None:
            if not cp.id or not cp.path:
                return
            cpath = cp.full_path.resolve().as_posix().lower()
            if check_path and cpath in seen_paths:
                return

            checkpoints.append(cp)
            seen_paths.add(cpath)

        # 1. Load checkpoints from YAML
        for cp_id, cp_data in raw_dict.items():
            if not isinstance(cp_data, dict):
                continue
            cp = Checkpoint().parse(cp_data, base_file, cp_id)
            add(cp, False)

        # 2. Discover additional local *.pt files strictly inside the checkpoints directory
        if base_file.exists() and base_file.is_dir():
            for f in sorted(base_file.glob("*.pt")):
                file_name = f.name
                file_rel = f.relative_to(base_file).as_posix()
                cp = Checkpoint(
                    id=file_name,
                    path=file_rel,
                    info=f"Local checkpoint ({file_name})",
                    base=base_file.as_posix(),
                )
                add(cp, True)

        self._list = checkpoints
        return self._list

    def list(self) -> List[Checkpoint]:
        """Returns the list of loaded Checkpoint instances (loads lazily if not yet loaded)."""
        if self._list is None:
            self.load()
        return self._list or []

    def clear(self) -> None:
        """Clears the cached checkpoints list."""
        self._list = None

    def find(self, index_or_id: Union[int, str]) -> Optional[Checkpoint]:
        """
        Finds a checkpoint by 1-based index (int or digit string) or by ID/path.
        Returns Checkpoint or None if not found.
        """
        checkpoints = self.list()

        # Check for 1-based integer index or digit string within range
        str_val = str(index_or_id).strip()
        if str_val.isdigit():
            idx = int(str_val)
            if 1 <= idx <= len(checkpoints):
                return checkpoints[idx - 1]
            else:
                return None

        # Lookup by ID, path, or filename/stem
        target = str_val
        target_name = Path(str_val).name
        target_stem = Path(str_val).stem

        for cp in checkpoints:
            candidates = {
                cp.id,
                cp.path,
                Path(cp.path).name,
                Path(cp.path).stem,
                cp.full_path.as_posix(),
                cp.full_path.name,
                cp.full_path.stem,
            }
            if (
                target in candidates
                or target_name in candidates
                or target_stem in candidates
            ):
                return cp

        return None


# Backward-compatible alias
CheckpointData = CheckpointStore
