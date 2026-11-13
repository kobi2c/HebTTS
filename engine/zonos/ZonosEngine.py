import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

# Ensure internal zonos source directory is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_ZONOS_SRC = _CURRENT_DIR / "zonos"
if str(_ZONOS_SRC) not in sys.path:
    sys.path.insert(0, str(_ZONOS_SRC))

from config import Config, SeedMode, Speaker, SpeakerSample
from ..base import VoiceEngine


class ZonosEngine(VoiceEngine):
    """
    Self-contained Zonos speech synthesis engine supporting Hebrew and multilingual models.
    Inherits from VoiceEngine, providing model loading, speaker embedding caching,
    conditioning dictionary assembly, and post-creation audio processing.
    """

    DEFAULT_MODEL_CHOICE: str = "notmax123/Zonos-Hebrew"
    DEFAULT_LANGUAGE: str = "he"
    SAMPLE_RATE: int = 44100

    def __init__(
        self,
        model_choice: str = DEFAULT_MODEL_CHOICE,
        device: Optional[Union[str, Any]] = None,
    ) -> None:
        self.default_model_choice: str = model_choice

        if device is not None:
            self._device = device
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available() and "cuda" in str(self._device).lower():
            self._device_name = torch.cuda.get_device_name(0)
        else:
            self._device_name = "CPU"

        # Model and conditioning cache
        self.model: Optional[Any] = None
        self.cached_model_choice: Optional[str] = None
        self._cached_speaker_path: Optional[str] = None
        self._cached_speaker_embedding: Optional[torch.Tensor] = None
        self._last_tokens_str: Optional[str] = None

    @property
    def device(self) -> torch.device:
        """Computation device used by the engine."""
        return self._device

    @property
    def device_name(self) -> str:
        """Human-readable device name."""
        return self._device_name

    @property
    def sample_rate(self) -> int:
        """Zonos native DAC autoencoder sampling rate is 44100 Hz."""
        return self.SAMPLE_RATE

    @property
    def last_tokens_str(self) -> Optional[str]:
        """Summary description from the last synthesis."""
        return self._last_tokens_str

    def load_model(
        self,
        model_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Pre-loads Zonos model on demand. Caches loaded model instance across calls.
        """
        target_model = model_choice or self.default_model_choice
        if self.model is not None and self.cached_model_choice == target_model:
            return self.model

        try:
            from zonos.model import Zonos
        except ImportError as e:
            raise ImportError(
                f"Failed to import Zonos model architecture: {e}. "
                "Ensure Zonos dependencies (transformers, etc.) are installed."
            ) from e

        self.model = Zonos.from_pretrained(target_model, device=self.device)
        self.cached_model_choice = target_model
        return self.model

    def get_speaker_embedding(
        self,
        speaker: Speaker,
        sample: Optional[SpeakerSample] = None,
    ) -> Optional[torch.Tensor]:
        """
        Computes or retrieves cached speaker embedding tensor for Zonos conditioning.
        """
        resolved_sample = sample if sample is not None else speaker.sample()
        prompt_audio = resolved_sample.path
        if not prompt_audio or not os.path.exists(prompt_audio):
            return None

        if self._cached_speaker_path == prompt_audio and self._cached_speaker_embedding is not None:
            return self._cached_speaker_embedding

        wav_np, sr = sf.read(prompt_audio)
        wav = torch.tensor(wav_np).float()
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav.t()

        model = self.load_model()
        spk_emb = model.make_speaker_embedding(wav, sr)

        try:
            param = next(model.parameters())
            target_dtype = getattr(param, "dtype", None)
            if target_dtype is not None and type(target_dtype).__name__ == "dtype":
                spk_emb = spk_emb.to(self.device, dtype=target_dtype)
            else:
                spk_emb = spk_emb.to(self.device)
        except Exception:
            spk_emb = spk_emb.to(self.device)

        self._cached_speaker_path = prompt_audio
        self._cached_speaker_embedding = spk_emb
        return spk_emb

    def synthesize_chunk(
        self,
        text: str,
        speaker_embedding: Optional[torch.Tensor],
        config: Config,
        progress_callback: Optional[Callable[[torch.Tensor, int, int], bool]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Synthesizes a single text segment using Zonos conditioning and codebook generation.
        """
        try:
            from zonos.conditioning import make_cond_dict
        except ImportError as e:
            raise ImportError(
                f"Failed to import Zonos conditioning: {e}. "
                "Ensure Zonos dependencies (transformers, etc.) are installed."
            ) from e

        target_model = kwargs.get("model_choice", config.model_choice)
        model = self.load_model(target_model)

        language = kwargs.get("language", config.language)
        fmax = float(kwargs.get("fmax", config.fmax))
        pitch_std = float(kwargs.get("pitch_std", config.pitch_std))
        speaking_rate = float(kwargs.get("speaking_rate", config.speaking_rate))
        dnsmos_ovrl = float(kwargs.get("dnsmos_ovrl", config.dnsmos_ovrl))
        speaker_noised = bool(kwargs.get("speaker_noised", config.speaker_noised))
        cfg_scale = float(kwargs.get("cfg_scale", config.cfg_scale))
        vq_val = float(kwargs.get("vq_score", config.vq_score))
        emotions_raw = kwargs.get("emotions", config.emotions)

        emotion_vals = [float(x) for x in emotions_raw] if emotions_raw else []
        emotion_sum = sum(emotion_vals)

        unconditional_keys = set(kwargs.get("unconditional_keys", ["vqscore_8", "dnsmos_ovrl"]))
        if speaker_embedding is None:
            unconditional_keys.add("speaker")

        if emotion_sum <= 0.0 or not emotion_vals:
            # If emotions are unconfigured or all zero, mark as unconditional to avoid NaN division
            unconditional_keys.add("emotion")
            emotion_tensor = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
        else:
            emotion_tensor = torch.tensor(emotion_vals, device=self.device)

        vq_tensor = torch.tensor([vq_val] * 8, device=self.device).unsqueeze(0)

        clean_text = text.replace("'", "").replace("’", "").replace("‘", "")

        cond_dict = make_cond_dict(
            text=clean_text,
            language=language,
            speaker=speaker_embedding,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            dnsmos_ovrl=dnsmos_ovrl,
            speaker_noised=speaker_noised,
            device=self.device,
            unconditional_keys=list(unconditional_keys),
        )
        conditioning = model.prepare_conditioning(cond_dict)

        # For Zonos, optimal autoregressive sampling uses min_p=0.1 (or top_k=0 if using VALL-E's top_k=50 default)
        cfg_top_k = getattr(config, "top_k", 50)
        top_k = int(kwargs.get("top_k", 0 if cfg_top_k == 50 else cfg_top_k))
        top_p = float(kwargs.get("top_p", getattr(config, "top_p", 0.0)))
        min_p = float(kwargs.get("min_p", getattr(config, "min_p", 0.1 if top_k == 0 else 0.0)))
        linear = float(kwargs.get("linear", getattr(config, "linear", 0.0)))
        confidence = float(kwargs.get("confidence", getattr(config, "confidence", 0.0)))
        quadratic = float(kwargs.get("quadratic", getattr(config, "quadratic", 0.0)))
        max_new_tokens = int(kwargs.get("max_new_tokens", 86 * 30))

        sampling_params = dict(
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            linear=linear,
            conf=confidence,
            quad=quadratic,
        )

        codes = model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=kwargs.get("audio_prefix_codes", None),
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params=sampling_params,
            callback=progress_callback,
        )

        wav_out = model.autoencoder.decode(codes)
        if hasattr(wav_out, "cpu"):
            wav_out = wav_out.cpu()
        if hasattr(wav_out, "detach"):
            wav_out = wav_out.detach()
        if hasattr(wav_out, "dim") and wav_out.dim() == 2 and hasattr(wav_out, "size") and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]
        if hasattr(wav_out, "squeeze"):
            wav_out = wav_out.squeeze()
        if hasattr(wav_out, "numpy"):
            out_np = wav_out.numpy()
        else:
            out_np = np.asarray(wav_out)
        return np.asarray(out_np, dtype=np.float32)

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
        Synthesizes text into a 1D float32 audio waveform using Zonos.
        """
        cfg = config or Config()

        # 1. Configure RNG seed
        if cfg.seed_mode == SeedMode.Enabled:
            resolved_seed = self.addSeed(None)
        else:
            resolved_seed = self.addSeed(cfg.seed)
        cfg.seed = resolved_seed
        torch.manual_seed(resolved_seed)

        # 2. Extract / retrieve speaker embedding
        speaker_emb = self.get_speaker_embedding(speaker, sample)

        # 3. Prepare text lines
        lines = self.prepare_text(text, cfg)
        if not lines:
            self._last_tokens_str = None
            return np.zeros(0, dtype=np.float32)

        pause_sec = float(cfg.pause_duration) if hasattr(cfg, "pause_duration") else 0.3
        silence_samples = int(self.sample_rate * pause_sec)
        silence = np.zeros(silence_samples, dtype=np.float32)
        segments: List[np.ndarray] = []

        total_steps = len(lines)
        for i, line in enumerate(lines):
            if progress_callback is not None:
                progress_callback(i, total_steps, f"Synthesizing line {i + 1}/{total_steps}...")

            audio_chunk = self.synthesize_chunk(
                text=line,
                speaker_embedding=speaker_emb,
                config=cfg,
                **kwargs,
            )
            if len(audio_chunk) > 0:
                segments.append(audio_chunk)
                if i < len(lines) - 1 and pause_sec > 0:
                    segments.append(silence)

        if not segments:
            self._last_tokens_str = None
            return np.zeros(0, dtype=np.float32)

        combined = np.concatenate(segments) if len(segments) > 1 else segments[0]
        self._last_tokens_str = f"Zonos [{cfg.model_choice}]: {len(lines)} line(s)"
        return self.post_create(combined, cfg)
