import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Checkpoint:
    """Represents a neural model checkpoint file."""
    id        : str
    path      : str
    info      : str = ""
    gdrive_id : Optional[str] = None
    base      : str = "checkpoints"

    @property
    def full_path(self) -> str:
        """Returns the complete filesystem path relative to the workspace root."""
        if self.base and not Path(self.path).is_absolute():
            norm_base = str(self.base).replace("\\", "/").rstrip("/")
            norm_path = self.path.replace("\\", "/")
            if norm_path == norm_base or norm_path.startswith(norm_base + "/"):
                return norm_path
            return str((Path(self.base) / self.path).as_posix())
        return self.path

    @property
    def exists(self) -> bool:
        """Returns True if the checkpoint file exists on disk."""
        return os.path.exists(self.full_path)

    def download(self, quiet: bool = False) -> str:
        """
        Downloads the checkpoint file from Google Drive if it does not exist on disk.
        Uses the gdown python package with fallback to gdown CLI.
        """
        if self.exists:
            return self.full_path

        if not self.gdrive_id:
            raise ValueError(f"Checkpoint '{self.id}' does not exist on disk and has no gdrive_id.")

        out_path = Path(self.full_path).resolve()
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

        return str(out_path)


class CheckpointData:
    """Manages discovery, caching, and querying of model checkpoints from checkpoints/checkpoints.yaml."""

    DEFAULT_DATA_PATH: str = "checkpoints/checkpoints.yaml"

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        self._main_path: str = self.find_path(path)
        self._load_path: str = self._main_path
        self._list: Optional[List[Checkpoint]] = None

    @property
    def data_path(self) -> str:
        """Returns the main path of the checkpoints configuration file."""
        return self._main_path

    @classmethod
    def find_path(cls, path: Union[str, Path, None] = None) -> str:
        """Finds the path to the checkpoints configuration file."""
        if path is not None:
            return str(path)
        return cls.DEFAULT_DATA_PATH

    def load(self, path: Union[str, Path, None] = None) -> List[Checkpoint]:
        """Loads checkpoints from YAML and discovers additional *.pt files in the checkpoints directory."""
        self._load_path = str(path) if path is not None else self._main_path
        yaml_file = Path(self._load_path)
        base_dir  = yaml_file.parent

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
        seen_ids: set = set()
        seen_paths: set = set()
        norm_base_dir = str(base_dir).replace("\\", "/").rstrip("/")

        # 1. Load checkpoints from YAML
        for cp_id, cp_data in raw_dict.items():
            if not isinstance(cp_data, dict):
                continue

            raw_path = str(cp_data.get("path", cp_id))
            norm_raw = raw_path.replace("\\", "/")
            if norm_raw.startswith(norm_base_dir + "/"):
                rel_path = norm_raw[len(norm_base_dir) + 1 :]
            else:
                rel_path = norm_raw

            description = str(cp_data.get("description", ""))
            gdrive_id = cp_data.get("gdrive_id")

            checkpoints.append(
                Checkpoint(
                    id=str(cp_id),
                    path=rel_path,
                    info=description,
                    gdrive_id=str(gdrive_id) if gdrive_id else None,
                    base=norm_base_dir,
                )
            )
            seen_ids.add(str(cp_id).lower())
            seen_paths.add(rel_path.lower())
            seen_paths.add(Path(rel_path).name.lower())

        # 2. Discover additional local *.pt files strictly inside the checkpoints directory
        if base_dir.exists() and base_dir.is_dir():
            for f in sorted(base_dir.glob("*.pt")):
                f_name = f.name
                rel_f = str(f.relative_to(base_dir).as_posix())
                if f_name.lower() not in seen_ids and rel_f.lower() not in seen_paths:
                    checkpoints.append(
                        Checkpoint(
                            id=f_name,
                            path=rel_f,
                            info=f"Local checkpoint ({f_name})",
                            base=norm_base_dir,
                        )
                    )
                    seen_ids.add(f_name.lower())
                    seen_paths.add(rel_f.lower())

        self._list = checkpoints
        return self._list

    def list(self) -> List[Checkpoint]:
        """Returns the list of loaded Checkpoint instances (loads lazily if not yet loaded)."""
        if self._list is None:
            self.load()
        return self._list or []

    def find(self, index_or_id: Union[int, str]) -> Optional[Checkpoint]:
        """
        Finds a checkpoint by 1-based index (int or digit string) or by ID/path.
        Returns Checkpoint or None if not found.
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

        # Lookup by ID or path (case-insensitive, normalized slashes)
        target = str_val.replace("\\", "/").lower()
        for cp in checkpoints:
            norm_cp_path = cp.path.replace("\\", "/").lower()
            norm_full_path = cp.full_path.replace("\\", "/").lower()
            if (
                cp.id.lower() == target
                or norm_cp_path == target
                or norm_full_path == target
                or Path(cp.path).name.lower() == target
            ):
                return cp

        return None

    def clear(self) -> None:
        """Clears the cached checkpoints list."""
        self._list = None
