import argparse
import os
import sys



MODELS = {
    1: {
        "id": "notmax123/Zonos-Hebrew",
        "description": "Hebrew (Hybrid)",
        "default_language": "he",
        "default_text": "שלום, זהו מבחן דיבור עבור מודל זונוס בעברית.",
    },
    2: {
        "id": "Zyphra/Zonos-v0.1-transformer",
        "description": "English / Multi (Transformer)",
        "default_language": "en-us",
        "default_text": "Zonos is an open-weight text-to-speech model by Zyphra.",
    },
    3: {
        "id": "Zyphra/Zonos-v0.1-hybrid",
        "description": "English / Multi (Hybrid)",
        "default_language": "en-us",
        "default_text": "Zonos is an open-weight text-to-speech model by Zyphra.",
    },
}


def print_model_list():
    print("Available models:")
    print("-" * 65)
    print(f"{'Number':<8} {'Model ID':<35} {'Description'}")
    print("-" * 65)
    for num, meta in MODELS.items():
        print(f"{num:<8} {meta['id']:<35} {meta['description']}")
    print("-" * 65)


def main():
    parser = argparse.ArgumentParser(description="Zonos TTS Sample Generator")
    parser.add_argument("--list", action="store_true", help="Display available model numbers and IDs")
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        choices=list(MODELS.keys()),
        help="Model number to use (default: 1)",
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g. 'he', 'en-us')")
    parser.add_argument(
        "--speaker-audio", type=str, default="assets/exampleaudio.mp3", help="Speaker audio file for cloning"
    )
    parser.add_argument("--output", type=str, default="sample.wav", help="Output wav file path (default: sample.wav)")
    args = parser.parse_args()

    if args.list:
        print_model_list()
        return

    import soundfile as sf
    import torch
    from zonos.conditioning import make_cond_dict
    from zonos.model import Zonos
    from zonos.utils import DEFAULT_DEVICE as device

    model_info = MODELS[args.model]
    model_id = model_info["id"]
    language = args.language or model_info["default_language"]

    if args.text is not None:
        text = args.text
    elif args.model == 1:
        if os.path.exists("sample_he.txt"):
            with open("sample_he.txt", "r", encoding="utf-8") as f:
                text = f.read().strip()
        else:
            text = model_info["default_text"]
    else:
        if os.path.exists("sample.txt"):
            with open("sample.txt", "r", encoding="utf-8") as f:
                text = f.read().strip()
        else:
            text = model_info["default_text"]

    print(f"Selected Model [{args.model}]: {model_id}")
    print(f"Language: {language}")
    print(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"Loading model on {device}...")
    model = Zonos.from_pretrained(model_id, device=device)

    wav_np, sampling_rate = sf.read(args.speaker_audio)
    wav = torch.tensor(wav_np).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.t()
    speaker = model.make_speaker_embedding(wav.to(device), sampling_rate)

    torch.manual_seed(421)
    cond_dict = make_cond_dict(text=text, speaker=speaker, language=language, device=device)
    conditioning = model.prepare_conditioning(cond_dict)

    print("Generating audio...")
    codes = model.generate(conditioning)

    wavs = model.autoencoder.decode(codes).cpu()
    sf.write(args.output, wavs[0].cpu().numpy().T, model.autoencoder.sampling_rate)
    print(f"Saved generated audio to: {args.output}")


if __name__ == "__main__":
    main()
