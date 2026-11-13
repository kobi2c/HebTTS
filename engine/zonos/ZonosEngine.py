import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

# Ensure internal zonos source directory is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_ZONOS_SRC = _CURRENT_DIR / "zonos"
if str(_ZONOS_SRC) not in sys.path:
    sys.path.insert(0, str(_ZONOS_SRC))

from config import Config, SeedMode, Speaker, SpeakerSample
from ..VoiceEngine import VoiceEngine
from ..EngineInfo import EngineInfo


class ZonosEngine(VoiceEngine):
    """
    Self-contained Zonos speech synthesis engine supporting Hebrew and multilingual models.
    Inherits from VoiceEngine, providing model loading, speaker embedding caching,
    conditioning dictionary assembly, and post-creation audio processing.
    """

    info: ClassVar[EngineInfo] = EngineInfo(
        id="zonos",
        name="Zonos (Zyphra)",
        description="Continuous-space hybrid architecture with DAC neural audio codec. Expressive synthesis at 44.1 kHz.",
        sample_rate=44100,
        languages=["he", "en-us", "en-gb", "fr-fr", "de", "es", "cmn", "ja"],
    )

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
    def is_loaded(self) -> bool:
        """Returns True if the Zonos model is currently loaded in memory."""
        return self.model is not None

    @property
    def loaded_checkpoint(self) -> Optional[str]:
        """Returns the identifier of the currently loaded model choice."""
        return self.cached_model_choice

    @property
    def sample_rate(self) -> int:
        """Zonos native DAC autoencoder sampling rate is 44100 Hz."""
        return self.SAMPLE_RATE

    @property
    def last_tokens_str(self) -> Optional[str]:
        """Summary description from the last synthesis."""
        return self._last_tokens_str

    def clear_last_tokens(self) -> None:
        """Resets token usage stats."""
        self._last_tokens_str = None

    def unload_model(self) -> None:
        """Unloads model weights from memory and clears accelerator caches."""
        self.model = None
        self.cached_model_choice = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_cache(self) -> None:
        """Clears cached speaker embeddings and state."""
        self._cached_speaker_path = None
        self._cached_speaker_embedding = None
        self._last_tokens_str = None

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

    def prepare_speaker(
        self,
        speaker: Speaker,
        sample: Optional[SpeakerSample] = None,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Prepares speaker embedding tensor prior to chunk synthesis."""
        return self.get_speaker_embedding(speaker, sample)

    def synthesize_chunk(
        self,
        text: str,
        speaker: Optional[Union[Speaker, torch.Tensor]] = None,
        sample: Optional[SpeakerSample] = None,
        config: Optional[Config] = None,
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

        cfg = config or Config()

        # Resolve speaker embedding tensor
        if isinstance(speaker, Speaker):
            speaker_embedding = self.prepare_speaker(speaker, sample, config=cfg)
        elif speaker is not None and (hasattr(speaker, "dim") or type(speaker).__name__ == "Tensor"):
            speaker_embedding = speaker
        else:
            speaker_embedding = kwargs.get("speaker_embedding", self._cached_speaker_embedding)

        target_model = kwargs.get("model_choice", cfg.model_choice)
        model = self.load_model(target_model)

        language = kwargs.get("language", cfg.language)
        fmax = float(kwargs.get("fmax", cfg.fmax))
        pitch_std = float(kwargs.get("pitch_std", cfg.pitch_std))
        speaking_rate = float(kwargs.get("speaking_rate", cfg.speaking_rate))
        dnsmos_ovrl = float(kwargs.get("dnsmos_ovrl", cfg.dnsmos_ovrl))
        speaker_noised = bool(kwargs.get("speaker_noised", cfg.speaker_noised))
        cfg_scale = float(kwargs.get("cfg_scale", cfg.cfg_scale))
        vq_val = float(kwargs.get("vq_score", cfg.vq_score))
        emotions_raw = kwargs.get("emotions", cfg.emotions)

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
        cfg_top_k = getattr(cfg, "top_k", 50)
        top_k = int(kwargs.get("top_k", 0 if cfg_top_k == 50 else cfg_top_k))
        top_p = float(kwargs.get("top_p", getattr(cfg, "top_p", 0.0)))
        min_p = float(kwargs.get("min_p", getattr(cfg, "min_p", 0.1 if top_k == 0 else 0.0)))
        linear = float(kwargs.get("linear", getattr(cfg, "linear", 0.0)))
        confidence = float(kwargs.get("confidence", getattr(cfg, "confidence", 0.0)))
        quadratic = float(kwargs.get("quadratic", getattr(cfg, "quadratic", 0.0)))
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

        self._last_tokens_str = f"Zonos [{target_model}]"
        return np.asarray(out_np, dtype=np.float32)

    def create_panel(self) -> Any:
        """Constructs and returns the Gradio EnginePanel for Zonos."""
        from .ZonosPanel import ZonosPanel
        return ZonosPanel()
