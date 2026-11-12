import random
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union
import numpy as np

from config import Config, NoiseFix, Speaker, SpeakerSample


class VoiceEngine(ABC):
    """
    Abstract base class for speech synthesis engines.
    Subclasses encapsulate model loading, tokenizer management, text preparation,
    neural audio synthesis, and audio post-processing.
    """

    @property
    def device(self) -> Any:
        """Computation device used by the engine (e.g. 'cuda:0' or 'cpu')."""
        return "cpu"

    @property
    def device_name(self) -> str:
        """Human-readable device name (e.g. 'NVIDIA GeForce RTX 3080' or 'CPU')."""
        return "CPU"

    @property
    def sample_rate(self) -> int:
        """Output audio sampling rate in Hz (default: 24000)."""
        return 24000

    @property
    def last_tokens_str(self) -> Optional[str]:
        """String representation of token statistics from the last synthesis (e.g. '195 -> 425 (+230)')."""
        return None

    def load_model(self, checkpoint: Optional[str] = None, **kwargs: Any) -> Any:
        """Pre-loads model weights or verifies readiness (no-op by default)."""
        pass

    def addSeed(self, seed: Optional[int] = None) -> int:
        """
        Configures the random seed for reproducible synthesis.
        If seed is None, generates a random non-negative integer.
        Returns the resolved seed.
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        return int(seed)

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

    def prepare_text(
        self,
        text: Union[str, List[str]],
        config: Optional[Config] = None,
    ) -> List[str]:
        """
        Prepares raw text into chunks not exceeding config.max_chunk_size.
        Splits by sentences (.!?), clauses (,;:), and words.
        """
        cfg = config or Config()
        max_chunk_size=cfg.max_chunk_size

        if isinstance(text, str):
            return self.split_text(text, max_chunk_size=max_chunk_size)
        elif isinstance(text, (list, tuple)):
            chunks: List[str] = []
            for item in text:
                chunks.extend(self.split_text(str(item), max_chunk_size=max_chunk_size))
            return chunks
        else:
            raw = str(text).strip()
            return self.split_text(raw, max_chunk_size=max_chunk_size) if raw else []

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
        processed = audio.copy()

        # Peak normalization if clipping
        max_val = np.abs(processed).max() if len(processed) > 0 else 0.0
        if max_val > 1.0:
            processed = processed / max_val

        # Leading noise trimming with safety padding
        if cfg.trim_noise:
            processed = NoiseFix.trim(processed, sr=self.sample_rate, pad_ms=cfg.pad_ms)

        # Micro fade-in and fade-out
        if cfg.fade_ms > 0:
            processed = NoiseFix.fade(
                processed,
                sr=self.sample_rate,
                fade_in_ms=cfg.fade_ms,
                fade_out_ms=cfg.fade_ms,
            )

        return processed

    @abstractmethod
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
        Synthesizes text into an audio waveform.

        Args:
            speaker: Speaker instance providing voice identity and reference audio/text.
            sample: Optional SpeakerSample override (e.g. for chained prompts). Defaults to speaker.sample().
            config: Optional Config instance for hyperparameters (seed, temperature, pause_duration, fade_ms, etc.).
            text: Text to synthesize (single string or list of text segments).
            progress_callback: Optional callback func(line_idx, total_lines, desc) for progress updates.
            **kwargs: Engine-specific parameters.

        Returns:
            np.ndarray: Audio waveform samples as a 1D float32 array.
        """
        raise NotImplementedError
