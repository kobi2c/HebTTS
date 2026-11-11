# Sets up soundfile audio IO fallback and suppresses warnings

import logging
import os
import warnings

# Silence torchvision image extension warning since TTS does not use vision IO
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Silence deprecated weight_norm FutureWarnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*weight_norm.*")

import torch
import torchaudio
import soundfile as sf


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


# In PyTorch >= 2.6, torch.load defaults weights_only=True, which breaks legacy checkpoints
# that store OmegaConf configs (e.g. audiocraft MBD checkpoints) unless weights_only=False.
_orig_torch_load = torch.load


def _torch_load(f, *args, **kwargs):
    if "weights_only" not in kwargs or kwargs["weights_only"] is None:
        kwargs["weights_only"] = False
    return _orig_torch_load(f, *args, **kwargs)


torch.load = _torch_load
if hasattr(torch, "serialization") and hasattr(torch.serialization, "load"):
    torch.serialization.load = _torch_load

try:
    import omegaconf
    import omegaconf.base
    import omegaconf.dictconfig
    import omegaconf.listconfig
    import omegaconf.nodes

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            omegaconf.dictconfig.DictConfig,
            omegaconf.listconfig.ListConfig,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Node,
            omegaconf.base.Metadata,
            omegaconf.nodes.ValueNode,
            omegaconf.nodes.AnyNode,
            omegaconf.nodes.StringNode,
            omegaconf.nodes.IntegerNode,
            omegaconf.nodes.BooleanNode,
            omegaconf.nodes.FloatNode,
            omegaconf.nodes.EnumNode,
        ])
except (ImportError, AttributeError):
    pass


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
