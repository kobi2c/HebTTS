import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio

from config.NoiseFix import NoiseFix
from openvoice import ToneColorConverter


class VoiceConverter:
    """
    Zero-shot Tone Color Converter for HebTTS.
    Converts a source voice recording (e.g., user speech) into the vocal timbre of a target speaker,
    preserving the source audio's exact pronunciation, timing, intonation, and rhythm.
    """
    CONFIG_URL = "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json"
    CHECKPOINT_URL = "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth"

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if checkpoint_dir is None:
            v2_dir = Path("checkpoints/converter_v2")
            v1_dir = Path("checkpoints/converter")
            if (v2_dir / "checkpoint.pth").exists():
                self.checkpoint_dir = v2_dir.resolve()
            elif (v1_dir / "checkpoint.pth").exists():
                self.checkpoint_dir = v1_dir.resolve()
            else:
                self.checkpoint_dir = v2_dir.resolve()
        else:
            self.checkpoint_dir = Path(checkpoint_dir).resolve()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._converter: Optional[ToneColorConverter] = None
        self._cached_se: Dict[str, torch.Tensor] = {}

    def ensure_checkpoints(self) -> Tuple[Path, Path]:
        """Downloads converter config and model checkpoint if not present."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.checkpoint_dir / "config.json"
        ckpt_path = self.checkpoint_dir / "checkpoint.pth"

        if not config_path.exists():
            print(f"Downloading voice converter config from {self.CONFIG_URL}...")
            urllib.request.urlretrieve(self.CONFIG_URL, str(config_path))

        if not ckpt_path.exists():
            print(f"Downloading voice converter weights (~125MB) from {self.CHECKPOINT_URL}...")
            urllib.request.urlretrieve(self.CHECKPOINT_URL, str(ckpt_path))
            print("Download complete!")

        return config_path, ckpt_path

    @property
    def converter(self) -> ToneColorConverter:
        """Lazy loads the ToneColorConverter model."""
        if self._converter is None:
            config_path, ckpt_path = self.ensure_checkpoints()
            dev_str = str(self.device)
            self._converter = ToneColorConverter(config_path, device=dev_str)
            self._converter.load_ckpt(ckpt_path)
        return self._converter

    def get_speaker_se(self, speaker_wav_path: Union[str, Path]) -> torch.Tensor:
        """Extracts and caches the target speaker's tone color embedding."""
        p_str = str(Path(speaker_wav_path).resolve())
        if p_str not in self._cached_se:
            se = self.converter.extract_se(p_str)
            self._cached_se[p_str] = se
        return self._cached_se[p_str]

    def convert(
        self,
        source_audio_path: Union[str, Path],
        target_speaker_path: Union[str, Path],
        output_path: Union[str, Path],
        target_sr: int = 24000,
        tau: float = 0.05,
        trim_noise: bool = True,
        pad_ms: float = 30.0,
        fade_ms: float = 20.0,
    ) -> Path:
        """
        Converts source speech audio to the target speaker's timbre.
        Applies input dynamic range normalization, RMS volume matching with target reference,
        resample to target_sr, and NoiseFix boundary smoothing.
        """
        src_p = Path(source_audio_path).resolve()
        tgt_p = Path(target_speaker_path).resolve()
        out_p = Path(output_path).resolve()

        if not src_p.exists():
            raise FileNotFoundError(f"Source audio file not found: {src_p}")
        if not tgt_p.exists():
            raise FileNotFoundError(f"Target speaker audio file not found: {tgt_p}")

        # Read and preprocess source audio for optimal dynamic range & clean SE extraction
        tmp_src_path = src_p
        try:
            wav_src, sr_src = sf.read(str(src_p), dtype="float32")
            if wav_src.ndim > 1:
                wav_src = np.mean(wav_src, axis=1)
            wav_src = wav_src - np.mean(wav_src)
            max_src = np.max(np.abs(wav_src))
            if max_src > 1e-4:
                wav_src = wav_src / max_src * 0.95

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                tmp_src_path = Path(tmp_f.name)
            sf.write(str(tmp_src_path), wav_src, sr_src)
        except Exception:
            tmp_src_path = src_p

        try:
            # Extract embeddings
            src_se = self.converter.extract_se(str(tmp_src_path))
            tgt_se = self.get_speaker_se(str(tgt_p))

            # Convert timbre with non-stochastic clean sampling
            converted_np = self.converter.convert(
                audio_src_path=str(tmp_src_path),
                src_se=src_se,
                tgt_se=tgt_se,
                tau=tau,
            )
        finally:
            if tmp_src_path != src_p and tmp_src_path.exists():
                tmp_src_path.unlink(missing_ok=True)

        model_sr = self.converter.hps.data.sampling_rate

        # Resample to target_sr if different
        if model_sr != target_sr:
            t = torch.tensor(converted_np, dtype=torch.float32).unsqueeze(0)
            resampled_t = torchaudio.functional.resample(t, model_sr, target_sr)
            converted_np = resampled_t.squeeze(0).cpu().numpy()

        # Match target reference RMS volume so converted chunk matches surrounding chunks
        try:
            wav_tgt, _ = sf.read(str(tgt_p), dtype="float32")
            if wav_tgt.ndim > 1:
                wav_tgt = np.mean(wav_tgt, axis=1)
            tgt_rms = np.sqrt(np.mean(wav_tgt**2))
            cur_rms = np.sqrt(np.mean(converted_np**2))
            if cur_rms > 1e-4 and tgt_rms > 1e-4:
                converted_np = converted_np * (tgt_rms / cur_rms)
        except Exception:
            pass

        max_val = np.abs(converted_np).max()
        if max_val > 0.95:
            converted_np = converted_np / max_val * 0.95

        # Apply NoiseFix trimming and fading
        if trim_noise:
            converted_np = NoiseFix.trim(converted_np, sr=target_sr, pad_ms=pad_ms)
        if fade_ms > 0:
            converted_np = NoiseFix.fade(converted_np, sr=target_sr, fade_in_ms=fade_ms, fade_out_ms=fade_ms)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), converted_np, target_sr)
        return out_p


def convert_chunk_voice(
    source_audio_path: Union[str, Path],
    target_speaker_path: Union[str, Path],
    output_path: Union[str, Path],
    target_sr: int = 24000,
    device: Optional[Union[str, torch.device]] = None,
) -> Path:
    """Convenience function for one-off voice conversion."""
    vc = VoiceConverter(device=device)
    return vc.convert(source_audio_path, target_speaker_path, output_path, target_sr=target_sr)
