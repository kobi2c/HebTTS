import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
nn = torch.nn

try:
    import librosa
except ImportError:
    librosa = None


class VoiceEncoder(nn.Module):
    """
    Generalized End-to-End (GE2E) Recurrent Voice Encoder for Speaker Verification.
    Maps audio mel-spectrogram frames to an L2-normalized 256-dimensional speaker d-vector.
    """
    def __init__(self, mel_n_channels: int = 40, model_hidden_size: int = 256, model_embedding_size: int = 256, num_layers: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of mel-spectrogram slices (batch_size, n_frames, n_channels).
        Returns unit-normalized d-vectors (batch_size, embedding_size).
        """
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)


class SpeakerSame:
    """
    Biometric Speaker Verification and Similarity Evaluator.
    Computes calibrated 0-100% certainty scores of whether two audio files belong to the same speaker.
    """
    WEIGHTS_URL = "https://raw.githubusercontent.com/resemble-ai/Resemblyzer/master/resemblyzer/pretrained.pt"
    SAMPLING_RATE = 16000
    MEL_CHANNELS = 40
    MEL_WINDOW_MS = 25
    MEL_STEP_MS = 10
    PARTIAL_FRAMES = 100  # 1.0 second per partial slice (dense temporal coverage for short & long audio)
    PARTIAL_STEP = 25     # 0.25 second step (75% overlap)

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        encoder: str = "resemblyzer",
    ) -> None:
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device_str = device
            if "cuda" in device_str and not torch.cuda.is_available():
                device_str = "cpu"
            self.device = torch.device(device_str)
        else:
            self.device = device

        self.weights_path = Path(weights_path) if weights_path else Path("checkpoints/speaker_encoder/pretrained.pt")
        self.encoder_name = encoder.lower()
        self._encoder: Optional[VoiceEncoder] = None
        self._wavlm_model = None
        self._wavlm_extractor = None

    def ensure_checkpoint(self) -> Path:
        """Downloads the pretrained 1.4MB speaker encoder checkpoint if missing."""
        resolved_p = self.weights_path.resolve()
        if not resolved_p.exists():
            resolved_p.parent.mkdir(parents=True, exist_ok=True)
            print(f"[SpeakerSame] Downloading speaker encoder weights (~1.4MB) from {self.WEIGHTS_URL}...")
            urllib.request.urlretrieve(self.WEIGHTS_URL, str(resolved_p))
            print("[SpeakerSame] Download complete!")
        return resolved_p

    @property
    def encoder(self) -> VoiceEncoder:
        """Lazy loads the VoiceEncoder neural network onto the selected device."""
        if self._encoder is None:
            ckpt_path = self.ensure_checkpoint()
            model = VoiceEncoder(
                mel_n_channels=self.MEL_CHANNELS,
                model_hidden_size=256,
                model_embedding_size=256,
                num_layers=3,
            )
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self._encoder = model
        return self._encoder

    def load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Loads and pre-processes audio to 16kHz mono, trimmed and volume-normalized."""
        p = Path(audio_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        wav, sr = sf.read(str(p), dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        # Resample to 16,000 Hz if necessary
        if sr != self.SAMPLING_RATE:
            if librosa is not None:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.SAMPLING_RATE)
            else:
                # Basic torchaudio fallback
                import torchaudio
                t = torch.tensor(wav).unsqueeze(0)
                wav = torchaudio.functional.resample(t, sr, self.SAMPLING_RATE).squeeze(0).numpy()

        # Simple energy silence trim
        if librosa is not None:
            trimmed, _ = librosa.effects.trim(wav, top_db=30)
            if len(trimmed) >= int(self.SAMPLING_RATE * 0.2):
                wav = trimmed

        # Normalize volume
        max_val = np.max(np.abs(wav))
        if max_val > 1e-4:
            wav = wav / max_val * 0.95

        return wav

    def wav_to_mel(self, wav: np.ndarray) -> np.ndarray:
        """Computes 40-band mel-spectrogram."""
        n_fft = int(self.SAMPLING_RATE * self.MEL_WINDOW_MS / 1000)
        hop_length = int(self.SAMPLING_RATE * self.MEL_STEP_MS / 1000)

        if librosa is not None:
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=self.SAMPLING_RATE,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.MEL_CHANNELS,
            )
            return mel.astype(np.float32).T
        else:
            import torchaudio
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.SAMPLING_RATE,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.MEL_CHANNELS,
            )
            t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
            mel = mel_transform(t).squeeze(0).numpy()
            return mel.astype(np.float32).T

    def embed_audio_wavlm(self, audio: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Computes a 512-D L2-normalized speaker embedding using WavLM-SV."""
        if self._wavlm_model is None:
            from transformers import AutoFeatureExtractor, WavLMForXVector
            model_id = "microsoft/wavlm-base-plus-sv"
            self._wavlm_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self._wavlm_model = WavLMForXVector.from_pretrained(model_id).to(self.device)
            self._wavlm_model.eval()

        if isinstance(audio, (str, Path)):
            wav = self.load_audio(audio)
        else:
            wav = audio

        inputs = self._wavlm_extractor(wav, sampling_rate=self.SAMPLING_RATE, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self._wavlm_model(**inputs).embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy()

    def embed_audio(self, audio: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Computes an L2-normalized speaker embedding from audio file or array."""
        if self.encoder_name == "wavlm":
            return self.embed_audio_wavlm(audio)

        if isinstance(audio, (str, Path)):
            wav = self.load_audio(audio)
        else:
            wav = audio

        mel = self.wav_to_mel(wav)
        n_frames = len(mel)

        if n_frames < self.PARTIAL_FRAMES:
            pad_len = self.PARTIAL_FRAMES - n_frames
            mel = np.pad(mel, ((0, pad_len), (0, 0)), mode="constant")
            slices = [mel]
        else:
            slices = [
                mel[i : i + self.PARTIAL_FRAMES]
                for i in range(0, n_frames - self.PARTIAL_FRAMES + 1, self.PARTIAL_STEP)
            ]
            # Ensure the tail slice of audio is included so ending syllables are not discarded
            if (n_frames - self.PARTIAL_FRAMES) % self.PARTIAL_STEP != 0:
                slices.append(mel[-self.PARTIAL_FRAMES :])
            if not slices:
                slices = [mel[: self.PARTIAL_FRAMES]]

        batch = torch.tensor(np.array(slices), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            partials = self.encoder(batch).cpu().numpy()

        raw_embed = np.mean(partials, axis=0)
        norm = np.linalg.norm(raw_embed)
        return raw_embed / (norm if norm > 1e-6 else 1.0)

    @staticmethod
    def calculate_certainty(
        cosine_similarity: float,
        midpoint: float = 0.76,
        steepness: float = 22.0,
        encoder: str = "resemblyzer",
    ) -> float:
        """
        Converts non-negative cosine similarity into a calibrated 0-100% confidence score.
        - Exactly identical files (cosine >= 0.9999) return 100.0%.
        - Scores >= 0.85 map to >= 85.0% certainty for Resemblyzer.
        - Midpoint (0.76) maps to ~50.0% certainty (equal error threshold).
        - Different speakers (typically 0.50-0.67) map to low certainty (< 15.0%).
        """
        cosine = float(np.clip(cosine_similarity, 0.0, 1.0))
        if cosine >= 0.9999:
            return 100.0

        if encoder.lower() == "wavlm":
            midpoint = 0.945
            steepness = 40.0
            min_bound = 0.80
        else:
            min_bound = 0.50

        # Calibrated normalized sigmoid: anchors min at min_bound and max at 1.00
        z = steepness * (cosine - midpoint)
        sig = 1.0 / (1.0 + np.exp(-z))

        sig_min = 1.0 / (1.0 + np.exp(-steepness * (min_bound - midpoint)))
        sig_max = 1.0 / (1.0 + np.exp(-steepness * (1.00 - midpoint)))

        norm_score = (sig - sig_min) / (sig_max - sig_min)
        return float(np.clip(norm_score * 100.0, 0.0, 100.0))

    def compare(
        self,
        speaker_audio: Union[str, Path, np.ndarray],
        test_audio: Union[str, Path, np.ndarray],
        threshold: float = 75.0,
    ) -> Dict[str, Any]:
        """
        Compares reference speaker audio and test audio.
        Returns a dictionary with certainty score (0-100), cosine similarity, and verdict.
        """
        emb_spk = self.embed_audio(speaker_audio)
        emb_test = self.embed_audio(test_audio)

        cosine = float(np.dot(emb_spk, emb_test))
        certainty = self.calculate_certainty(cosine, encoder=self.encoder_name)
        is_same = bool(certainty >= threshold)

        return {
            "certainty": round(certainty, 1),
            "cosine_similarity": round(cosine, 4),
            "is_same": is_same,
            "threshold": float(threshold),
            "encoder": self.encoder_name,
            "speaker": str(speaker_audio) if isinstance(speaker_audio, (str, Path)) else "array",
            "test": str(test_audio) if isinstance(test_audio, (str, Path)) else "array",
        }


def speaker_same(
    speaker_audio: Union[str, Path],
    test_audio: Union[str, Path],
    threshold: float = 75.0,
    device: Optional[Union[str, torch.device]] = None,
    encoder: str = "resemblyzer",
) -> Tuple[bool, float, float]:
    """
    Convenience function: compares speaker audio and test audio.
    Returns (is_same_speaker: bool, certainty_score: float, cosine_similarity: float).
    """
    verifier = SpeakerSame(device=device, encoder=encoder)
    res = verifier.compare(speaker_audio, test_audio, threshold=threshold)
    return res["is_same"], res["certainty"], res["cosine_similarity"]
