from pathlib import Path
from typing import Optional, Union
import librosa
import numpy as np
import soundfile
import torch

from openvoice import utils
from openvoice.mel_processing import spectrogram_torch
from openvoice.models import SynthesizerTrn


class ToneColorConverter:
    """
    Zero-shot Tone Color Converter based on OpenVoice VITS architecture.
    Extracts speaker embeddings directly from spectrograms and transfers vocal timbre.
    """
    def __init__(self, config_path: Union[str, Path], device: Union[str, torch.device] = "cuda:0") -> None:
        device_str = str(device)
        if "cuda" in device_str and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)

        hps = utils.get_hparams_from_file(str(config_path))
        model = SynthesizerTrn(
            len(getattr(hps, "symbols", [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(self.device)
        model.eval()

        self.model = model
        self.hps = hps

    def load_ckpt(self, ckpt_path: Union[str, Path]) -> None:
        checkpoint_dict = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint_dict["model"], strict=False)

    def extract_se(self, ref_wav_path: Union[str, Path]) -> torch.Tensor:
        """Extracts speaker tone color embedding tensor (1, D, 1) from reference audio."""
        audio_ref, _ = librosa.load(str(ref_wav_path), sr=self.hps.data.sampling_rate)
        y = torch.FloatTensor(audio_ref).to(self.device).unsqueeze(0)
        spec = spectrogram_torch(
            y,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        ).to(self.device)
        with torch.no_grad():
            g = self.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
        return g.detach()

    def convert(
        self,
        audio_src_path: Union[str, Path],
        src_se: torch.Tensor,
        tgt_se: torch.Tensor,
        output_path: Optional[Union[str, Path]] = None,
        tau: float = 0.3,
    ) -> np.ndarray:
        """Converts source speech audio to target speaker timbre."""
        audio, _ = librosa.load(str(audio_src_path), sr=self.hps.data.sampling_rate)
        y = torch.FloatTensor(audio).to(self.device).unsqueeze(0)
        with torch.no_grad():
            spec = spectrogram_torch(
                y,
                self.hps.data.filter_length,
                self.hps.data.sampling_rate,
                self.hps.data.hop_length,
                self.hps.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            converted = self.model.voice_conversion(
                spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau
            )[0][0, 0].data.cpu().float().numpy()

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(str(out_p), converted, self.hps.data.sampling_rate)
        return converted

