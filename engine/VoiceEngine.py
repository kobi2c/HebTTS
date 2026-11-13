import random
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union
import numpy as np

from config import Config, NoiseFix, SeedMode, Speaker, SpeakerSample
from .EngineInfo import EngineInfo


class VoiceEngine(ABC):
    """
    Abstract base class defining the speech synthesis engine contract.
    Subclasses encapsulate engine metadata, model loading/unloading, speaker preparation,
    chunk-level neural audio synthesis, and audio post-processing.
    """

    info: ClassVar[EngineInfo] = EngineInfo(
        id="base",
        name="Base Voice Engine",
        description="",
        sample_rate=24000,
        languages=["he"],
    )

    @property
    def device(self) -> Any:
        """Computation device used by the engine (e.g. 'cuda:0' or 'cpu')."""
        return "cpu"

    @property
    def device_name(self) -> str:
        """Human-readable device name (e.g. 'NVIDIA GeForce RTX 3080' or 'CPU')."""
        return "CPU"

    @property
    def is_loaded(self) -> bool:
        """Indicates whether neural model weights are currently resident in memory."""
        return False

    @property
    def loaded_checkpoint(self) -> Optional[str]:
        """Identifier or path of the currently resident checkpoint or model variant."""
        return None

    @property
    def last_tokens_str(self) -> Optional[str]:
        """String representation of token statistics from the last synthesis (e.g. '195 -> 425 (+230)')."""
        return None

    def live_info(self) -> EngineInfo:
        """Returns an EngineInfo merging static metadata with current runtime state."""
        return EngineInfo(
            id=self.info.id,
            name=self.info.name,
            description=self.info.description,
            sample_rate=self.info.sample_rate,
            device=str(self.device),
            device_name=self.device_name,
            is_loaded=self.is_loaded,
            loaded_checkpoint=self.loaded_checkpoint,
            languages=list(self.info.languages),
            last_tokens_str=self.last_tokens_str,
        )

    # --- Model & Cache Lifecycle ---

    def load_model(self, checkpoint: Optional[str] = None, **kwargs: Any) -> Any:
        """Pre-loads model weights or verifies readiness (no-op by default)."""
        pass

    def unload_model(self) -> None:
        """Unloads model weights from memory and clears accelerator caches."""
        pass

    def clear_cache(self) -> None:
        """Clears cached speaker embeddings, audio prompts, and intermediate buffers."""
        pass

    def clear_last_tokens(self) -> None:
        """Resets token usage string when no audio is produced."""
        pass

    # --- RNG & Seed Management ---

    def apply_seed(
        self,
        seed: Optional[int] = None,
        mode: Union[SeedMode, int, bool] = SeedMode.Enabled,
    ) -> int:
        """
        Configures the random seed for reproducible synthesis across Python, NumPy, and PyTorch.
        Returns the resolved seed.
        """
        if int(mode) == int(SeedMode.Enabled) or seed is None:
            resolved = random.randint(0, 2**31 - 1)
        else:
            resolved = int(seed)

        random.seed(resolved)
        np.random.seed(resolved % (2**32 - 1))
        try:
            import torch
            torch.manual_seed(resolved)
        except Exception:
            pass

        return resolved

    # --- Text Preparation & Splitting ---

    @staticmethod
    def split_text(text: str, max_chunk_size: int = 100) -> List[str]:
        """
        Splits text into natural clauses and sentences not exceeding max_chunk_size characters.
        Preserves line breaks and sentence context to prevent attention overflow.
        """
        all_combined_parts: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("---") or line.startswith("["):
                continue
            # Clean problematic punctuation / dashes
            line = line.replace("_", " ")
            line = line.replace("—", "-").replace("–", "-")
            line = line.replace("’", "'").replace("‘", "'")
            # 1. Split by major sentence boundaries (.!?)
            sentences = re.split(r'(?<=[.!?])\s+', line)
            parts: List[str] = []
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(s) <= max_chunk_size:
                    parts.append(s)
                    continue
                # 2. Split by secondary clause boundaries (,;:)
                sub_sentences = re.split(r'(?<=[,;:])\s+', s)
                current_part = ""
                for sub in sub_sentences:
                    sub = sub.strip()
                    if not sub:
                        continue
                    if len(sub) > max_chunk_size:
                        # 3. If still exceeding max_chunk_size, split by words
                        words = sub.split(" ")
                        for w in words:
                            if len(current_part) + len(w) + 1 > max_chunk_size:
                                if current_part:
                                    parts.append(current_part.strip())
                                current_part = w + " "
                            else:
                                current_part += w + " "
                    else:
                        if len(current_part) + len(sub) + 1 > max_chunk_size:
                            if current_part:
                                parts.append(current_part.strip())
                            current_part = sub + " "
                        else:
                            current_part += sub + " "
                if current_part:
                    parts.append(current_part.strip())
            # Combine small parts up to max_chunk_size
            combined_parts: List[str] = []
            current_part = ""
            for p in parts:
                if len(current_part) + len(p) + 1 <= max_chunk_size:
                    current_part += (" " if current_part else "") + p
                else:
                    if current_part:
                        combined_parts.append(current_part)
                    current_part = p
            if current_part:
                combined_parts.append(current_part)
            all_combined_parts.extend(combined_parts)
        return all_combined_parts

    def preprocess_text(self, text: str, config: Optional[Config] = None) -> str:
        """
        Engine-specific hook for text normalization (e.g. vocalization, phonetic conversion).
        Defaults to returning the input text unchanged.
        """
        return text

    def prepare_text(
        self,
        text: Union[str, List[str]],
        config: Optional[Config] = None,
    ) -> List[str]:
        """
        Prepares raw text into non-empty chunks not exceeding config.max_chunk_size,
        applying preprocess_text normalization to each chunk.
        """
        cfg = config or Config()
        max_chunk_size = cfg.max_chunk_size

        if isinstance(text, str):
            raw_lines = self.split_text(text, max_chunk_size=max_chunk_size)
        elif isinstance(text, (list, tuple)):
            raw_lines = []
            for item in text:
                raw_lines.extend(self.split_text(str(item), max_chunk_size=max_chunk_size))
        else:
            raw = str(text).strip()
            raw_lines = self.split_text(raw, max_chunk_size=max_chunk_size) if raw else []

        prepared: List[str] = []
        for line in raw_lines:
            processed = self.preprocess_text(line, cfg).strip()
            if processed:
                prepared.append(processed)
        return prepared

    # --- Speaker Preparation Hook ---

    def prepare_speaker(
        self,
        speaker: Speaker,
        sample: Optional[SpeakerSample] = None,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Hook to compute or cache speaker prompt tokens or embeddings once prior to chunk synthesis.
        Returns the cached prompt representation or None.
        """
        return None

    # --- Audio Post-Processing ---

    def post_create(
        self,
        audio: np.ndarray,
        config: Optional[Config] = None,
    ) -> np.ndarray:
        """
        Applies post-synthesis audio processing including peak normalization,
        leading noise trimming, and edge micro-fading based on Config.
        """
        if len(audio) == 0:
            return audio

        cfg = config or Config()
        processed = np.asarray(audio, dtype=np.float32).copy()

        # Peak normalization if clipping
        max_val = np.abs(processed).max() if len(processed) > 0 else 0.0
        if max_val > 1.0:
            processed = processed / max_val

        # Leading noise trimming with safety padding
        if cfg.trim_noise:
            processed = NoiseFix.trim(processed, sr=self.info.sample_rate, pad_ms=cfg.pad_ms)

        # Micro fade-in and fade-out
        if cfg.fade_ms > 0:
            processed = NoiseFix.fade(
                processed,
                sr=self.info.sample_rate,
                fade_in_ms=cfg.fade_ms,
                fade_out_ms=cfg.fade_ms,
            )

        return np.asarray(processed, dtype=np.float32)

    # --- Abstract Chunk Synthesis ---

    @abstractmethod
    def synthesize_chunk(
        self,
        text: str,
        speaker: Speaker,
        sample: Optional[SpeakerSample] = None,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Synthesizes a single chunk of text into a 1D float32 numpy audio waveform.

        Args:
            text: Single chunk of text to synthesize.
            speaker: Speaker instance providing voice identity.
            sample: Resolved SpeakerSample reference.
            config: Configuration hyperparameters.
            **kwargs: Engine-specific parameters.

        Returns:
            np.ndarray: 1D float32 audio waveform.
        """
        raise NotImplementedError

    @abstractmethod
    def create_panel(self) -> Any:
        """Constructs and returns the Gradio EnginePanel for this engine."""
        raise NotImplementedError

    # --- Template Method: create ---

    def create(
        self,
        speaker: Speaker,
        sample: Optional[SpeakerSample] = None,
        config: Optional[Config] = None,
        text: Union[str, List[str]] = "",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Template method orchestrating the complete text-to-speech synthesis pipeline:
        1. Configures RNG seeds.
        2. Splits and preprocesses text into chunks.
        3. Prepares reference speaker representation once.
        4. Synthesizes each chunk, interleaving inter-sentence pause silence.
        5. Concatenates chunks and applies post-synthesis normalization/fading.

        Args:
            speaker: Speaker instance providing voice identity and reference audio/text.
            sample: Optional SpeakerSample override. Defaults to speaker.sample().
            config: Optional Config instance for hyperparameters.
            text: Text to synthesize (single string or list of text segments).
            progress_callback: Optional callback func(chunk_idx, total_chunks, desc).
            **kwargs: Engine-specific parameters forwarded to synthesize_chunk().

        Returns:
            np.ndarray: Audio waveform samples as a 1D float32 array.
        """
        cfg = config or Config()

        # 1. Configure RNG seed
        cfg.seed = self.apply_seed(cfg.seed, cfg.seed_mode)

        # 2. Text preparation
        lines = self.prepare_text(text, cfg)
        if not lines:
            self.clear_last_tokens()
            return np.zeros(0, dtype=np.float32)

        # 3. Resolve speaker sample and prepare speaker embedding/prompts
        resolved_sample = sample if sample is not None else speaker.sample()
        self.prepare_speaker(speaker, resolved_sample, config=cfg, **kwargs)

        # 4. Multi-chunk synthesis with pause interleaving
        pause_sec = float(cfg.pause_duration) if hasattr(cfg, "pause_duration") else 0.3
        silence_samples = int(self.info.sample_rate * pause_sec)
        silence = np.zeros(silence_samples, dtype=np.float32) if (pause_sec > 0 and len(lines) > 1) else None

        segments: List[np.ndarray] = []
        total_lines = len(lines)
        for i, line in enumerate(lines):
            if progress_callback is not None:
                progress_callback(i, total_lines, f"Synthesizing line {i + 1}/{total_lines}...")

            chunk_audio = self.synthesize_chunk(
                text=line,
                speaker=speaker,
                sample=resolved_sample,
                config=cfg,
                **kwargs,
            )
            if len(chunk_audio) > 0:
                segments.append(chunk_audio)
                if silence is not None and i < total_lines - 1:
                    segments.append(silence)

        if not segments:
            self.clear_last_tokens()
            return np.zeros(0, dtype=np.float32)

        combined = np.concatenate(segments) if len(segments) > 1 else segments[0]
        return self.post_create(combined, cfg)
