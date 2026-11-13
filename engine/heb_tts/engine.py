import os
import random
from pathlib import Path
from typing import Any, Callable, ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from config import AttributeDict, CheckpointStore, Config, NikudMaker, NoiseFix, SeedMode, Speaker, SpeakerSample
from engine.heb_tts.valle.data import AudioTokenizer, tokenize_audio
from engine.heb_tts.valle.data.collation import get_text_token_collater
from engine.heb_tts.valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars
from engine.heb_tts.valle.models import get_model
from ..base import VoiceEngine


class HebTTSEngine(VoiceEngine):
    """
    Self-contained Hebrew VALL-E speech synthesis engine.
    Caches model weights, vocoders, and prompt audio embeddings for optimal performance.
    """

    engine_id: ClassVar[str] = "hebtts"
    name: ClassVar[str] = "HebTTS (VALL-E)"
    description: ClassVar[str] = "Hebrew VALL-E neural codec language model. Fast autoregressive synthesis at 24 kHz."
    default_sample_rate: ClassVar[int] = 24000
    supported_languages: ClassVar[List[str]] = ["he"]

    PACKAGE_DIR = Path(__file__).resolve().parent
    DEFAULT_TOKENS_FILE = str(PACKAGE_DIR / "tokenizer" / "unique_words_tokens_all.k2symbols")
    DEFAULT_VOCAB_FILE = str(PACKAGE_DIR / "tokenizer" / "vocab.txt")

    def __init__(
        self,
        tokens_file: str = DEFAULT_TOKENS_FILE,
        vocab_file: str = DEFAULT_VOCAB_FILE,
        device: Optional[Union[str, Any]] = None,
    ) -> None:
        if device is not None:
            self._device = device
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available() and "cuda" in str(self._device).lower():
            self._device_name = torch.cuda.get_device_name(0)
        else:
            self._device_name = "CPU"

        self.tokens_file: str = tokens_file
        self.vocab_file: str = vocab_file
        self.nikud_maker: NikudMaker = NikudMaker()

        # Model and tokenizer cache
        self.model: Optional[torch.nn.Module] = None
        self.audio_tokenizer: Optional[AudioTokenizer] = None
        self.text_collater: Optional[Any] = None
        self.alef_bert_tokenizer: Optional[AlefBERTRootTokenizer] = None
        self.cached_checkpoint: Optional[str] = None
        self.cached_mbd: Optional[bool] = None

        # Audio prompt cache
        self._cached_prompt_audio: Optional[str] = None
        self._cached_prompt_text: Optional[str] = None
        self._cached_audio_prompts: Optional[torch.Tensor] = None
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
        """Returns True if the VALL-E model is loaded in memory."""
        return self.model is not None

    @property
    def loaded_checkpoint(self) -> Optional[str]:
        """Returns the path of the currently loaded checkpoint."""
        return self.cached_checkpoint

    @property
    def last_tokens_str(self) -> Optional[str]:
        """Token statistics from the last synthesis."""
        return self._last_tokens_str

    def clear_last_tokens(self) -> None:
        """Resets token usage stats."""
        self._last_tokens_str = None

    def unload_model(self) -> None:
        """Unloads model weights and vocoder from memory and clears GPU cache."""
        self.model = None
        self.audio_tokenizer = None
        self.text_collater = None
        self.alef_bert_tokenizer = None
        self.cached_checkpoint = None
        self.cached_mbd = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_cache(self) -> None:
        """Clears cached speaker audio prompts and token stats."""
        self._cached_prompt_audio = None
        self._cached_prompt_text = None
        self._cached_audio_prompts = None
        self._last_tokens_str = None

    def load_model(
        self,
        checkpoint: Optional[str] = None,
        mbd: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.nn.Module, AudioTokenizer, Any, AlefBERTRootTokenizer]:
        """
        Pre-loads VALL-E model and vocoder on demand. Caches loaded models across calls.
        """
        target_cp = checkpoint or "checkpoint_01.pt"
        cp_entry = CheckpointStore().find(target_cp)
        if cp_entry:
            if not cp_entry.exists and cp_entry.gdrive_id:
                cp_entry.download()
            actual_path = str(cp_entry.full_path)
        else:
            actual_path = target_cp

        if (
            self.model is not None
            and self.cached_checkpoint == actual_path
            and self.cached_mbd == bool(mbd)
        ):
            return self.model, self.audio_tokenizer, self.text_collater, self.alef_bert_tokenizer

        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"Checkpoint file '{actual_path}' does not exist.")

        checkpoint_data = torch.load(actual_path, map_location=self.device, weights_only=False)
        args = AttributeDict(checkpoint_data)
        model = get_model(args)
        missing_keys, _ = model.load_state_dict(checkpoint_data["model"], strict=True)
        assert not missing_keys, f"Missing model keys: {missing_keys}"
        model.to(self.device)
        model.eval()
        model.verbose = False

        self.model = model
        self.text_collater = get_text_token_collater(self.tokens_file)
        self.audio_tokenizer = AudioTokenizer(mbd=bool(mbd), device=self.device)
        self.alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=self.vocab_file)
        self.cached_checkpoint = actual_path
        self.cached_mbd = bool(mbd)

        return self.model, self.audio_tokenizer, self.text_collater, self.alef_bert_tokenizer

    def preprocess_text(self, text: str, config: Optional[Config] = None) -> str:
        """Converts Hebrew vocalized words into phonetic Ktiv-Male if nikud_mode is enabled."""
        cfg = config or Config()
        if cfg.nikud_mode:
            return self.nikud_maker.convert(text)
        return text

    def prepare_speaker(
        self,
        speaker: Speaker,
        sample: Optional[SpeakerSample] = None,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Tokenizes and caches speaker audio prompts prior to chunk synthesis."""
        resolved_sample = sample if sample is not None else speaker.sample()
        prompt_audio = resolved_sample.path
        prompt_text = resolved_sample.text
        cfg = config or Config()
        if cfg.nikud_mode:
            prompt_text = self.nikud_maker.convert(prompt_text)

        # Ensure model and tokenizers are initialized
        model_tuple = self.load_model(cfg.checkpoint, cfg.mbd)
        if isinstance(model_tuple, (tuple, list)) and len(model_tuple) >= 4:
            if self.model is None and model_tuple[0] is not None:
                self.model = model_tuple[0]
            if self.audio_tokenizer is None and model_tuple[1] is not None:
                self.audio_tokenizer = model_tuple[1]
            if self.text_collater is None and model_tuple[2] is not None:
                self.text_collater = model_tuple[2]
            if self.alef_bert_tokenizer is None and model_tuple[3] is not None:
                self.alef_bert_tokenizer = model_tuple[3]

        if self._cached_prompt_audio != prompt_audio or self._cached_audio_prompts is None:
            encoded_frames = tokenize_audio(self.audio_tokenizer, prompt_audio)
            self._cached_audio_prompts = (
                torch.concat([encoded_frames[0][0]], dim=-1).transpose(2, 1).to(self.device)
            )
            self._cached_prompt_audio = prompt_audio
            self._cached_prompt_text = prompt_text

        return self._cached_audio_prompts

    def synthesize_chunk(
        self,
        text: str,
        speaker: Speaker,
        sample: Optional[SpeakerSample] = None,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Synthesizes a single chunk of Hebrew text using the loaded model and vocoders.
        Returns a 1D float32 numpy audio waveform.
        """
        cfg = config or Config()
        audio_prompts = self.prepare_speaker(speaker, sample, config=cfg)
        prompt_text = self._cached_prompt_text or (sample.text if sample else speaker.sample().text)

        if cfg.nikud_mode:
            text = self.preprocess_text(text, cfg)
            prompt_text = self.preprocess_text(prompt_text, cfg)

        text_without_space = [replace_chars(f"{prompt_text} {text}").strip().replace(" ", "_")]
        tokens = self.alef_bert_tokenizer._tokenize(text_without_space)
        prompt_text_without_space = [replace_chars(f"{prompt_text}").strip().replace(" ", "_")]
        prompt_tokens = self.alef_bert_tokenizer._tokenize(prompt_text_without_space)

        # Filter out unsupported tokens
        tokens = [t for t in tokens if t in self.text_collater.token2idx]
        prompt_tokens = [t for t in prompt_tokens if t in self.text_collater.token2idx]

        if not tokens:
            return np.zeros(0, dtype=np.float32)

        text_tokens, text_tokens_lens = self.text_collater([tokens])
        _, enroll_x_lens = self.text_collater([prompt_tokens])

        top_k = int(kwargs.get("top_k", cfg.top_k))
        temperature = float(kwargs.get("temperature", cfg.temperature))
        mbd = bool(int(kwargs.get("mbd", cfg.mbd)))

        with torch.no_grad():
            encoded_frames = self.model.inference(
                text_tokens.to(self.device),
                text_tokens_lens.to(self.device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=top_k,
                temperature=temperature,
            )

            is_cuda = "cuda" in str(self.device).lower()
            if mbd and getattr(self.audio_tokenizer, "mbd", None) is not None:
                if is_cuda and torch.cuda.is_available():
                    self.model.to("cpu")
                    torch.cuda.empty_cache()
                samples = self.audio_tokenizer.mbd_decode(encoded_frames.transpose(2, 1))
                if is_cuda and torch.cuda.is_available():
                    self.model.to(self.device)
            else:
                samples = self.audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])

        # Record token usage statistics
        eos = getattr(self.model, "last_eos", None)
        if eos and isinstance(eos, (tuple, list)) and len(eos) >= 2:
            p_len, tot_len = eos[0], eos[1]
            diff = tot_len - p_len
            self._last_tokens_str = f"{p_len} -> {tot_len} (+{diff})"

        seg_tensor = samples[0].cpu()
        if hasattr(seg_tensor, "shape") and seg_tensor.shape[-1] == 0:
            return np.zeros(0, dtype=np.float32)

        if hasattr(seg_tensor, "squeeze"):
            seg_tensor = seg_tensor.squeeze()
        if hasattr(seg_tensor, "numpy"):
            wav_np = seg_tensor.numpy()
        else:
            wav_np = np.asarray(seg_tensor)

        return np.asarray(wav_np, dtype=np.float32)
