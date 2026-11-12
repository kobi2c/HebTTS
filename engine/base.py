import random
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

    def prepare_text(
        self,
        text: Union[str, List[str]],
        config: Optional[Config] = None,
    ) -> List[str]:
        """
        Prepares raw text into non-empty, stripped lines.
        Subclasses can extend this to apply language-specific normalization.
        """
        if isinstance(text, str):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
        elif isinstance(text, (list, tuple)):
            lines = []
            for item in text:
                for line in str(item).splitlines():
                    cleaned = line.strip()
                    if cleaned:
                        lines.append(cleaned)
        else:
            lines = [str(text).strip()] if str(text).strip() else []
        return lines

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
