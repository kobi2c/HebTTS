import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable
import numpy as np

from config import Speaker, Config, SpeakerSample


class VoiceEngine(ABC):
    """
    Abstract base class for speech synthesis engines.
    Subclasses encapsulate model loading, tokenizer management, and audio synthesis.
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

    def seed(self, seed: Optional[int] = None) -> int:
        """Alias for addSeed."""
        return self.addSeed(seed)

    @abstractmethod
    def create(
        self,
        speaker: Speaker,
        text: str,
        config: Optional[Config] = None,
        prompt_sample: Optional[SpeakerSample] = None,
        fade_ms: float = 20.0,
        trim_noise: bool = True,
        pad_ms: float = 30.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Synthesizes a text segment into an audio waveform.

        Args:
            speaker: Speaker instance providing voice identity and reference audio/text.
            text: Text to synthesize (single or multi-line).
            config: Optional Config instance for hyperparameters (seed, temperature, pause_duration, etc.).
            prompt_sample: Optional SpeakerSample override (e.g. for chained prompts).
            fade_ms: Micro fade duration in milliseconds.
            trim_noise: Whether to trim leading vocoder startup noise.
            pad_ms: Pre-speech safety silence padding in milliseconds.
            progress_callback: Optional callback func(line_idx, total_lines, desc) for progress updates.
            **kwargs: Engine-specific parameters.

        Returns:
            np.ndarray: Audio waveform samples as a 1D float array.
        """
        raise NotImplementedError

