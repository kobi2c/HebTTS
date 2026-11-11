import os
import random
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from config import CheckpointStore, Config, NoiseFix, RandomMode, Speaker, SpeakerSample
from nikud_converter import convert_nikud_text
from utils import AttributeDict
from valle.data import AudioTokenizer, tokenize_audio
from valle.data.collation import get_text_token_collater
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars
from valle.models import get_model
from .base import VoiceEngine


class HebTTSEngine(VoiceEngine):
    """
    Self-contained Hebrew VALL-E speech synthesis engine.
    Caches model weights, vocoders, and prompt audio embeddings for optimal performance.
    """

    DEFAULT_TOKENS_FILE = "valle/data/tokens.txt"
    DEFAULT_VOCAB_FILE = "valle/data/vocab.txt"

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
    def sample_rate(self) -> int:
        """HebTTS native sampling rate is 24000 Hz."""
        return 24000

    @property
    def last_tokens_str(self) -> Optional[str]:
        """Token statistics from the last synthesis."""
        return self._last_tokens_str

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

    def synthesize_line(
        self,
        text: str,
        audio_prompts: torch.Tensor,
        prompt_text: str,
        top_k: int = 50,
        temperature: float = 1.0,
        mbd: bool = False,
        nikud_converter: bool = True,
    ) -> torch.Tensor:
        """Synthesizes a single line of Hebrew text using the loaded model and tokenizers."""
        if nikud_converter:
            text = convert_nikud_text(text)
            prompt_text = convert_nikud_text(prompt_text)

        text_without_space = [replace_chars(f"{prompt_text} {text}").strip().replace(" ", "_")]
        tokens = self.alef_bert_tokenizer._tokenize(text_without_space)
        prompt_text_without_space = [replace_chars(f"{prompt_text}").strip().replace(" ", "_")]
        prompt_tokens = self.alef_bert_tokenizer._tokenize(prompt_text_without_space)

        # Filter out any unsupported characters/tokens not present in the vocabulary
        tokens = [t for t in tokens if t in self.text_collater.token2idx]
        prompt_tokens = [t for t in prompt_tokens if t in self.text_collater.token2idx]

        if not tokens:
            return torch.zeros((1, 0))

        text_tokens, text_tokens_lens = self.text_collater([tokens])
        _, enroll_x_lens = self.text_collater([prompt_tokens])

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
            if mbd and is_cuda and torch.cuda.is_available():
                self.model.to("cpu")

            samples = self.audio_tokenizer.decode([(encoded_frames.transpose(2, 1), enroll_x_lens)])

            if mbd and is_cuda and torch.cuda.is_available():
                self.model.to(self.device)

        return samples[0].cpu()

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
        Synthesizes Hebrew text (single or multi-line) into a 1D float32 audio waveform.
        """
        cfg = config or Config()

        # 1. Resolve prompt audio and text
        sample = prompt_sample if prompt_sample is not None else speaker.sample()
        prompt_audio = sample.path
        prompt_text = sample.text

        # 2. Configure RNG seed
        if cfg.randomize_seed == RandomMode.Enabled:
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = int(cfg.seed_number)
        torch.manual_seed(seed)

        # 3. Load model, vocoder, and tokenizers
        model, audio_tokenizer, text_collater, alef_bert_tokenizer = self.load_model(
            cfg.checkpoint, cfg.mbd
        )
        model.verbose = False
        model.last_eos = None

        # 4. Cache audio prompt frames if reference changed
        if self._cached_prompt_audio != prompt_audio or self._cached_audio_prompts is None:
            encoded_frames = tokenize_audio(audio_tokenizer, prompt_audio)
            self._cached_audio_prompts = (
                torch.concat([encoded_frames[0][0]], dim=-1).transpose(2, 1).to(self.device)
            )
            self._cached_prompt_audio = prompt_audio
            self._cached_prompt_text = prompt_text

        # 5. Split input text into lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            self._last_tokens_str = None
            return np.zeros(0, dtype=np.float32)

        pause_sec = float(cfg.pause_duration) if hasattr(cfg, "pause_duration") else 0.3
        silence = torch.zeros((1, int(self.sample_rate * pause_sec)))
        segments: List[torch.Tensor] = []

        for i, line in enumerate(lines):
            if progress_callback is not None:
                progress_callback(i, len(lines), f"Synthesizing line {i + 1}/{len(lines)}...")

            seg = self.synthesize_line(
                text=line,
                audio_prompts=self._cached_audio_prompts,
                prompt_text=self._cached_prompt_text or prompt_text,
                top_k=int(cfg.top_k),
                temperature=float(cfg.temperature),
                mbd=bool(cfg.mbd),
                nikud_converter=bool(cfg.nikud_converter),
            )
            if seg.shape[-1] > 0:
                segments.append(seg)
                if i < len(lines) - 1 and pause_sec > 0:
                    segments.append(silence)

        if not segments:
            self._last_tokens_str = None
            return np.zeros(0, dtype=np.float32)

        full_audio = torch.cat(segments, dim=-1)
        audio_np = full_audio.squeeze().cpu().numpy()

        max_val = np.abs(audio_np).max() if len(audio_np) > 0 else 0.0
        if max_val > 1.0:
            audio_np = audio_np / max_val

        # 6. Apply post-processing
        if trim_noise:
            audio_np = NoiseFix.trim(audio_np, sr=self.sample_rate, pad_ms=pad_ms)
        if fade_ms > 0:
            audio_np = NoiseFix.fade(audio_np, sr=self.sample_rate, fade_in_ms=fade_ms, fade_out_ms=fade_ms)

        # 7. Record token usage statistics
        eos = getattr(model, "last_eos", None)
        if eos and isinstance(eos, (tuple, list)) and len(eos) >= 2:
            p_len, tot_len = eos[0], eos[1]
            diff = tot_len - p_len
            self._last_tokens_str = f"{p_len} -> {tot_len} (+{diff})"
        else:
            self._last_tokens_str = None

        return audio_np
