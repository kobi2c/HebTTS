import os
import warnings
import torch
import torchaudio
import soundfile as sf

# Silence torchvision image extension warning since TTS does not use vision IO
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def _soundfile_load(uri, *args, **kwargs):
    data, sr = sf.read(uri, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sr


def _soundfile_save(uri, src, sample_rate, *args, format=None, **kwargs):
    tensor = src.detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor.t()
    path_str = str(uri)
    ext = os.path.splitext(path_str)[1].lower()
    if not ext:
        if format is None:
            format = "WAV"
        if not path_str.endswith(".wav"):
            path_str += ".wav"
            uri = path_str
    sf.write(uri, tensor.numpy(), sample_rate, format=format)


try:
    import torchcodec
except ImportError:
    torchaudio.load = _soundfile_load
    torchaudio.save = _soundfile_save


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")
