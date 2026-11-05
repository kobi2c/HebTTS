import argparse
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

import utils  # Sets up soundfile audio IO fallback and suppresses torchvision warning
import torch
import torchaudio

from valle.data import AudioTokenizer, tokenize_audio
from valle.data.collation import get_text_token_collater
from valle.models import get_model
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars
from utils import AttributeDict
from nikud_converter import convert_nikud_text


class Sampler:
    CHECKPOINTS = {
        1: {
            "id": "checkpoint.pt",
            "description": "VALL-E Hebrew (AR 1.2M / NAR 200K steps on HebDB)",
            "path": "checkpoint.pt",
            "gdrive_id": "11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv",
        }
    }

    DEFAULT_SPEAKER_YAML = "speakers/speakers.yaml"
    DEFAULT_TOKENS_FILE  = "tokenizer/unique_words_tokens_all.k2symbols"
    DEFAULT_VOCAB_FILE   = "tokenizer/vocab.txt"

    def __init__(
        self,
        speaker_yaml=DEFAULT_SPEAKER_YAML,
        tokens_file=DEFAULT_TOKENS_FILE,
        vocab_file=DEFAULT_VOCAB_FILE,
    ):
        self.speaker_yaml = speaker_yaml
        self.tokens_file = tokens_file
        self.vocab_file = vocab_file

    def get_speakers_dict(self, yaml_path=None):
        path = yaml_path or self.speaker_yaml
        if not os.path.exists(path):
            return {}
        speakers_conf = OmegaConf.load(path)
        return OmegaConf.to_container(speakers_conf, resolve=True)

    def print_speaker_list(self, speakers=None):
        if speakers is None:
            speakers = self.get_speakers_dict()
        print("\nAvailable speakers:")
        print("-" * 80)
        print(f"{'Number':<8} {'Speaker ID':<15} {'Audio Prompt':<15} {'Text Prompt'}")
        print("-" * 80)
        for idx, (spk_id, spk_data) in enumerate(speakers.items(), start=1):
            prompt_text = spk_data.get("text-prompt", "")
            if len(prompt_text) > 35:
                prompt_text = prompt_text[:32] + "..."
            audio_prompt = spk_data.get("audio-prompt", "")
            print(f"{idx:<8} {spk_id:<15} {audio_prompt:<15} {prompt_text}")
        print("-" * 80 + "\n")

    def print_checkpoint_list(self):
        print("\nAvailable checkpoints:")
        print("-" * 85)
        print(f"{'Number':<8} {'Checkpoint ID':<20} {'Status':<12} {'Description'}")
        print("-" * 85)
        for num, meta in self.CHECKPOINTS.items():
            exists = os.path.exists(meta["path"])
            status = "Found" if exists else "Missing"
            print(f"{num:<8} {meta['id']:<20} {status:<12} {meta['description']}")
        print("-" * 85 + "\n")

    def resolve_checkpoint(self, checkpoint_arg):
        if checkpoint_arg.isdigit():
            idx = int(checkpoint_arg)
            if idx in self.CHECKPOINTS:
                return self.CHECKPOINTS[idx]["path"]
            else:
                print(f"Error: Predefined checkpoint number {idx} not found.")
                self.print_checkpoint_list()
                sys.exit(1)
        return checkpoint_arg

    def resolve_speaker(self, speaker_arg, speakers=None):
        if speakers is None:
            speakers = self.get_speakers_dict()

        if speaker_arg.isdigit():
            idx = int(speaker_arg)
            keys = list(speakers.keys())
            if 1 <= idx <= len(keys):
                spk_id = keys[idx - 1]
                return spk_id, speakers[spk_id]
            else:
                print(f"Error: Speaker number {idx} not found. Must be between 1 and {len(keys)}.")
                self.print_speaker_list(speakers)
                sys.exit(1)

        if speaker_arg in speakers:
            return speaker_arg, speakers[speaker_arg]

        print(f"Error: Speaker '{speaker_arg}' not found.")
        self.print_speaker_list(speakers)
        sys.exit(1)

    def load_model(self, checkpoint, device):
        if not checkpoint or not os.path.exists(checkpoint):
            print(f"Error: Checkpoint file '{checkpoint}' does not exist.")
            print("To download the official checkpoint, run:")
            print("  docker run --rm -v %CD%:/app hebtts:latest gdown 11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv")
            sys.exit(1)

        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        args = AttributeDict(checkpoint_data)
        model = get_model(args)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_data["model"], strict=True)
        assert not missing_keys, f"Missing model keys: {missing_keys}"
        model.to(device)
        model.eval()
        return model, args.text_tokens_path

    def synthesize_line(
        self,
        text,
        model,
        text_collater,
        audio_tokenizer,
        alef_bert_tokenizer,
        audio_prompts,
        prompt_text,
        device,
        top_k=50,
        temperature=1.0,
        mbd=False,
        nikud_converter=True,
    ):
        if nikud_converter:
            text = convert_nikud_text(text)
            prompt_text = convert_nikud_text(prompt_text)

        text_without_space = [replace_chars(f"{prompt_text} {text}").strip().replace(" ", "_")]
        tokens = alef_bert_tokenizer._tokenize(text_without_space)
        prompt_text_without_space = [replace_chars(f"{prompt_text}").strip().replace(" ", "_")]
        prompt_tokens = alef_bert_tokenizer._tokenize(prompt_text_without_space)

        # Filter out any unsupported characters/tokens not present in the vocabulary
        tokens = [t for t in tokens if t in text_collater.token2idx]
        prompt_tokens = [t for t in prompt_tokens if t in text_collater.token2idx]

        if not tokens:
            return torch.zeros((1, 0))

        text_tokens, text_tokens_lens = text_collater([tokens])
        _, enroll_x_lens = text_collater([prompt_tokens])

        with torch.no_grad():
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=top_k,
                temperature=temperature,
            )

            if mbd:
                samples = audio_tokenizer.mbd_decode(encoded_frames.transpose(2, 1))
            else:
                samples = audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])

        return samples[0].cpu()

    def synthesize(
        self,
        lines,
        model,
        text_collater,
        audio_tokenizer,
        alef_bert_tokenizer,
        audio_prompts,
        prompt_text,
        device,
        top_k=50,
        temperature=1.0,
        mbd=False,
        silence_duration_s=0.3,
        nikud_converter=True,
    ):
        if isinstance(lines, str):
            lines = [lines]

        if len(lines) == 1:
            print(f"Text:                {lines[0][:60]}{'...' if len(lines[0]) > 60 else ''}")
            print("Generating audio...")
            return self.synthesize_line(
                lines[0],
                model,
                text_collater,
                audio_tokenizer,
                alef_bert_tokenizer,
                audio_prompts,
                prompt_text,
                device,
                top_k=top_k,
                temperature=temperature,
                mbd=mbd,
                nikud_converter=nikud_converter,
            )

        print(f"Found {len(lines)} lines in input. Generating audio...")
        silence = torch.zeros((1, int(24000 * silence_duration_s)))
        segments = []
        for i, line in enumerate(lines, start=1):
            print(f"[{i}/{len(lines)}] {line[:55]}{'...' if len(line) > 55 else ''}")
            seg = self.synthesize_line(
                line,
                model,
                text_collater,
                audio_tokenizer,
                alef_bert_tokenizer,
                audio_prompts,
                prompt_text,
                device,
                top_k=top_k,
                temperature=temperature,
                mbd=mbd,
                nikud_converter=nikud_converter,
            )
            if seg.shape[-1] > 0:
                segments.append(seg)
                if i < len(lines):
                    segments.append(silence)
        return torch.cat(segments, dim=-1) if segments else torch.zeros((1, 0))

    def create_parser(self):
        parser = argparse.ArgumentParser(description="HebTTS Speech Generator")
        parser.add_argument(
            "--speaker-list",
            action="store_true",
            help="Display available speakers",
        )
        parser.add_argument(
            "--checkpoint-list",
            action="store_true",
            help="Display predefined checkpoints",
        )
        parser.add_argument(
            "--speaker",
            type=str,
            default="1",
            help="Speaker name or number (default: 1)",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="1",
            help="Checkpoint number or path (default: 1 -> checkpoint.pt)",
        )
        parser.add_argument(
            "--input",
            "--in-file",
            type=str,
            default=None,
            dest="input_file",
            help="Input text file path (output will default to same directory and stem with .wav)",
        )
        parser.add_argument(
            "--text",
            type=str,
            default=None,
            help="Hebrew text to synthesize",
        )
        parser.add_argument(
            "--clone-wav",
            "--speaker-audio",
            type=str,
            default=None,
            dest="clone_wav",
            help="Reference audio file path for voice cloning (.wav)",
        )
        parser.add_argument(
            "--clone-txt",
            "--speaker-text",
            type=str,
            default=None,
            dest="clone_txt",
            help="Reference text file path whose contents provide the clone transcript",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output wav file (default: same directory and stem as input file or sample.wav)",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=50,
            help="Top-k sampling parameter (default: 50)",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="Sampling temperature (default: 1.0)",
        )
        parser.add_argument(
            "--mbd",
            action="store_true",
            help="Enable Multi-Band Diffusion vocoder",
        )
        parser.add_argument(
            "--nikud-converter",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable automatic phonetic Nikud / Ktiv-Male disambiguation (default: True)",
        )
        return parser

    def run(self, args=None):
        if args is None:
            args = self.create_parser().parse_args()
        elif isinstance(args, list):
            args = self.create_parser().parse_args(args)

        speakers = self.get_speakers_dict()

        if args.speaker_list:
            self.print_speaker_list(speakers)
            return

        if args.checkpoint_list:
            self.print_checkpoint_list()
            return

        checkpoint_path = self.resolve_checkpoint(args.checkpoint)
        speaker_id, speaker_data = self.resolve_speaker(args.speaker, speakers)

        # Resolve text input
        if args.input_file is not None:
            if not os.path.exists(args.input_file):
                print(f"Error: Input file '{args.input_file}' not found.")
                sys.exit(1)
            with open(args.input_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.read().splitlines() if line.strip()]
        elif args.text is not None:
            lines = [args.text.strip()]
        elif os.path.exists("sample_he.txt"):
            with open("sample_he.txt", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.read().splitlines() if line.strip()]
        elif os.path.exists("sample.txt"):
            with open("sample.txt", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.read().splitlines() if line.strip()]
        else:
            lines = ["שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית."]

        if not lines:
            lines = ["שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית."]

        # Resolve output path: if not explicitly specified, default to same directory and stem as input file
        if args.output is not None:
            output_path = Path(args.output)
        elif args.input_file is not None:
            output_path = Path(args.input_file).with_suffix(".wav")
        else:
            output_path = Path("sample.wav")

        if not output_path.suffix:
            output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve prompt audio and text
        if args.clone_wav:
            prompt_audio = args.clone_wav
            if not os.path.exists(prompt_audio):
                print(f"Error: Clone audio file '{prompt_audio}' not found.")
                sys.exit(1)

            if args.clone_txt:
                if os.path.exists(args.clone_txt):
                    with open(args.clone_txt, "r", encoding="utf-8") as f:
                        prompt_text = f.read().strip()
                elif "\n" in args.clone_txt or " " in args.clone_txt:
                    prompt_text = args.clone_txt.strip()
                else:
                    print(f"Error: Clone text file '{args.clone_txt}' not found.")
                    sys.exit(1)
            else:
                auto_txt = Path(prompt_audio).with_suffix(".txt")
                if auto_txt.exists():
                    with open(auto_txt, "r", encoding="utf-8") as f:
                        prompt_text = f.read().strip()
                else:
                    print(f"Error: Clone text file not specified and '{auto_txt}' not found.")
                    print("Please specify reference text with --clone-txt <file.txt>.")
                    sys.exit(1)
        else:
            prompt_audio = str(Path(self.speaker_yaml).parent / speaker_data["audio-prompt"])
            prompt_text = speaker_data["text-prompt"]

        # Device detection
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        print(f"Selected Checkpoint: {checkpoint_path}")
        if args.clone_wav:
            print(f"Clone Audio:         {prompt_audio}")
            print(f"Clone Text:          {prompt_text[:60]}{'...' if len(prompt_text) > 60 else ''}")
        else:
            print(f"Selected Speaker:    {speaker_id}")
        print(f"Device:              {device} ({device_name})")
        if args.input_file:
            print(f"Input File:          {args.input_file}")
        print("Loading model...")

        model, _ = self.load_model(checkpoint_path, device)
        text_collater = get_text_token_collater(self.tokens_file)
        audio_tokenizer = AudioTokenizer(mbd=args.mbd)
        alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=self.vocab_file)

        # Encode prompt audio
        encoded_frames = tokenize_audio(audio_tokenizer, prompt_audio)
        audio_prompts = torch.concat([encoded_frames[0][0]], dim=-1).transpose(2, 1).to(device)

        # Generate speech
        samples = self.synthesize(
            lines=lines,
            model=model,
            text_collater=text_collater,
            audio_tokenizer=audio_tokenizer,
            alef_bert_tokenizer=alef_bert_tokenizer,
            audio_prompts=audio_prompts,
            prompt_text=prompt_text,
            device=device,
            top_k=args.top_k,
            temperature=args.temperature,
            mbd=args.mbd,
            nikud_converter=args.nikud_converter,
        )

        torchaudio.save(str(output_path), samples, 24000)
        print(f"Saved generated audio to: {output_path}")
        return output_path


if __name__ == "__main__":
    Sampler().run()
