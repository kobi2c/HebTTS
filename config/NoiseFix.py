from typing import Optional
import numpy as np


def apply_fade(
    audio: np.ndarray,
    sr: int = 24000,
    fade_in_ms: float = 20.0,
    fade_out_ms: float = 20.0,
) -> np.ndarray:
    """Applies a smooth raised-cosine (Hann) fade-in and fade-out to suppress vocoder boundary clicks/noise."""
    if len(audio) == 0 or (fade_in_ms <= 0 and fade_out_ms <= 0):
        return audio
    audio = audio.copy()
    in_samples = min(int(sr * (fade_in_ms / 1000.0)), len(audio) // 2)
    out_samples = min(int(sr * (fade_out_ms / 1000.0)), len(audio) // 2)
    if in_samples > 0:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, in_samples, endpoint=True, dtype=audio.dtype)))
        audio[:in_samples] *= fade_in
    if out_samples > 0:
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, out_samples, endpoint=True, dtype=audio.dtype)))
        audio[-out_samples:] *= fade_out
    return audio


def trim_leading_noise(
    audio: np.ndarray,
    sr: int = 24000,
    pad_ms: float = 30.0,
    max_search_sec: float = 1.5,
) -> np.ndarray:
    """
    Trims leading vocoder startup noise and clicks from synthesized speech chunk.
    Detects the true speech onset and cuts at the silence valley preceding speech,
    preserving pad_ms of natural pre-speech silence.
    """
    if len(audio) == 0:
        return audio

    audio_flat = audio.squeeze() if audio.ndim > 1 else audio
    max_peak = float(np.max(np.abs(audio_flat)))
    if max_peak < 0.01:
        return audio  # Essentially silent audio

    # Adaptive speech detection thresholds based on overall audio peak
    speech_peak_thresh = max(0.045, 0.12 * max_peak)
    speech_rms_thresh = max(0.015, 0.04 * max_peak)

    win = int(sr * 0.01)    # 10ms analysis window
    step = int(sr * 0.005)  # 5ms step
    max_search = min(len(audio_flat) - win, int(sr * max_search_sec))

    onset_idx = None

    # Step 1: Detect sustained speech onset
    for i in range(0, max_search, step):
        chunk = audio_flat[i : i + win]
        pk = np.max(np.abs(chunk))
        rms = np.sqrt(np.mean(chunk**2))

        if pk >= speech_peak_thresh or rms >= speech_rms_thresh:
            # Confirm with lookahead that energy is sustained
            lookahead = audio_flat[i : min(len(audio_flat), i + int(sr * 0.05))]
            if np.max(np.abs(lookahead)) >= speech_peak_thresh:
                onset_idx = i
                break

    if onset_idx is None:
        return audio

    # Step 2: Search backwards for the quietest point (silence valley) before speech onset
    pad_samples = int(sr * (pad_ms / 1000.0))
    search_start = max(0, onset_idx - int(sr * 0.25))
    search_end = max(0, onset_idx - pad_samples)

    cut_idx = search_end
    if search_end > search_start:
        min_rms = float("inf")
        best_pos = search_end
        for pos in range(search_start, search_end, step):
            sub_chunk = audio_flat[pos : pos + win]
            sub_rms = np.sqrt(np.mean(sub_chunk**2))
            if sub_rms < min_rms:
                min_rms = sub_rms
                best_pos = pos
        cut_idx = best_pos

    # Return audio trimmed from the silence valley
    return audio[cut_idx:].copy()


class NoiseFix:
    """Helper class providing leading noise trimming and boundary fading utilities."""
    trim = staticmethod(trim_leading_noise)
    fade = staticmethod(apply_fade)
