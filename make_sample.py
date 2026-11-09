import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import utils  # Sets up soundfile audio IO fallback and suppresses torchvision warning
import torch
import torchaudio

from valle.data import AudioTokenizer, tokenize_audio
from valle.data.collation import get_text_token_collater
from valle.models import get_model
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars
from utils import AttributeDict
from nikud_converter import convert_nikud_text
from config import SpeakerStore, CheckpointStore, Format

__all__ = ["SampleMaker", "SampleMakerParser", "SampleMake", "Sampler", "SamplerParser"]


class SampleMakerParser(argparse.ArgumentParser):
    """Command line argument parser for SampleMaker."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            description="HebTTS Speech Generator",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            **kwargs,
        )
        self.add_argument(
            "--speaker-list",
            action="store_true",
            help="Display available speakers",
        )
        self.add_argument(
            "--checkpoint-list",
            action="store_true",
            help="Display predefined checkpoints",
        )
        self.add_argument(
            "--speaker",
            type=str,
            default="1",
            help="Speaker name or number (default: 1)",
        )
        self.add_argument(
            "--checkpoint",
            type=str,
            default="1",
            help="Checkpoint number or path (default: 1 -> checkpoint.pt)",
        )
        self.add_argument(
            "--input",
            type=str,
            default=None,
            dest="input_file",
            help="Input text file path (output will default to same directory and stem with .wav)",
        )
        self.add_argument(
            "--text",
            type=str,
            default=None,
            help="Hebrew text to synthesize",
        )
        self.add_argument(
            "--clone-wav", "--clone-audio",
            type=str,
            default=None,
            dest="clone_wav",
            help="Reference audio file path for voice cloning (.wav, .mp3)",
        )
        self.add_argument(
            "--clone-txt",
            type=str,
            default=None,
            dest="clone_txt",
            help="Reference text file path whose contents provide the clone transcript",
        )
        self.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output audio file (default: same directory and stem as input file or sample.wav)",
        )
        self.add_argument(
            "--format",
            type=str,
            choices=["wav", "mp3", "both"],
            default=None,
            help="Output audio format ('wav', 'mp3', or 'both')",
        )
        self.add_argument(
            "--top-k",
            type=int,
            default=50,
            help="Top-k sampling parameter (default: 50)",
        )
        self.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="Sampling temperature (default: 1.0)",
        )
        self.add_argument(
            "--mbd",
            action="store_true",
            help="Enable Multi-Band Diffusion vocoder",
        )
        self.add_argument(
            "--nikud-converter",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable automatic phonetic Nikud / Ktiv-Male disambiguation (default: True)",
        )


class SampleMaker:
    DEFAULT_TOKENS_FILE  = "tokenizer/unique_words_tokens_all.k2symbols"
    DEFAULT_VOCAB_FILE   = "tokenizer/vocab.txt"

    def __init__(
        self,
        tokens_file: str = DEFAULT_TOKENS_FILE,
        vocab_file: str = DEFAULT_VOCAB_FILE,
        speaker: str = "1",
        checkpoint: str = "1",
        input_file: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        clone_wav: Optional[str] = None,
        clone_txt: Optional[str] = None,
        output: Optional[Union[str, Path]] = None,
        format: Optional[List[Format]] = None,
        top_k: int = 50,
        temperature: float = 1.0,
        mbd: bool = False,
        nikud_converter: bool = True,
        speaker_list: bool = False,
        checkpoint_list: bool = False,
    ) -> None:
        self.speakers: SpeakerStore = SpeakerStore()
        self.checkpoints: CheckpointStore = CheckpointStore()
        self.tokens_file: str = tokens_file
        self.vocab_file: str = vocab_file
        self.speaker: str = speaker
        self.checkpoint: str = checkpoint
        self.input_file: Optional[Union[str, Path]] = input_file
        self.text: Optional[str] = text
        self.clone_wav: Optional[str] = clone_wav
        self.clone_txt: Optional[str] = clone_txt
        self.output: Optional[Union[str, Path]] = output
        if format is not None:
            self.format: List[Format] = Format.normalize(format)
        elif output is not None and str(output).lower().endswith(".mp3"):
            self.format: List[Format] = [Format.MP3]
        else:
            self.format: List[Format] = [Format.WAV]
        self.top_k: int = top_k
        self.temperature: float = temperature
        self.mbd: bool = mbd
        self.nikud_converter: bool = nikud_converter
        self.speaker_list: bool = speaker_list
        self.checkpoint_list: bool = checkpoint_list

    normalize_formats = Format.normalize



    def _print_table(self, title: str, items: list, fields: list):
        """Prints a formatted tabular list of items based on field definitions."""
        print(f"\n{title}")
        header_cols = [f"{'Number':<8}"] + [
            f"{f['title']:<{f['width']}}" if f["width"] > 0 else f["title"]
            for f in fields
        ]
        header = " ".join(header_cols)
        sep = "-" * max(len(header), 95)

        print(sep)
        print(header)
        print(sep)
        for idx, item in enumerate(items, start=1):
            row_cols = [f"{idx:<8}"] + [
                f"{str(f['value'](item)):<{f['width']}}" if f["width"] > 0 else str(f['value'](item))
                for f in fields
            ]
            print(" ".join(row_cols))
        print(sep + "\n")

    def print_speakers(self, speakers=None):
        if speakers is None:
            speakers = self.speakers.list()
        fields = [
            {"title": "ID"         ,  "width": 15, "value": lambda spk: spk.id},
            {"title": "Audio-File" ,  "width": 15, "value": lambda spk: spk.wav},
            {"title": "Audio-Text" ,  "width": 0,  "value": lambda spk: spk.txt[:32] + "..." if len(spk.txt) > 35 else spk.txt},
        ]
        self._print_table("Available speakers:", speakers, fields)

    def print_checkpoints(self):
        checkpoints = self.checkpoints.list()
        fields = [
            {"title": "Checkpoint ID", "width": 20, "value": lambda cp: cp.id},
            {"title": "Path",          "width": 20, "value": lambda cp: cp.path},
            {"title": "Status",        "width": 10, "value": lambda cp: "Found" if cp.exists else "Missing"},
            {"title": "Description",   "width": 0,  "value": lambda cp: cp.info},
        ]
        self._print_table("Available checkpoints:", checkpoints, fields)

    def resolve_checkpoint(self, checkpoint_arg):
        cp = self.checkpoints.find(checkpoint_arg)
        if cp:
            if not cp.exists and cp.gdrive_id:
                cp.download()
            return cp.full_path
        if os.path.exists(checkpoint_arg):
            return checkpoint_arg
        print(f"Error: Predefined checkpoint '{checkpoint_arg}' not found.")
        self.print_checkpoints()
        sys.exit(1)

    def resolve_speaker(self, speaker_arg, speakers=None):
        spk = self.speakers.find(speaker_arg)
        if spk:
            return spk.id, {"text-prompt": spk.txt, "audio-prompt": str(spk.full_path)}

        print(f"Error: Speaker '{speaker_arg}' not found.")
        self.print_speakers(speakers)
        sys.exit(1)

    def load_model(self, checkpoint, device):
        if not checkpoint or not os.path.exists(checkpoint):
            print(f"Error: Checkpoint file '{checkpoint}' does not exist.")
            print("To download the official checkpoint, run:")
            print("  docker-run gdown 11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv")
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

    def create_parser(self) -> SampleMakerParser:
        """Creates and returns the command line argument parser for SampleMaker."""
        return SampleMakerParser()

    def parse_args(
        self,
        args: Optional[Union[argparse.Namespace, List[str]]] = None,
    ) -> argparse.Namespace:
        """Parses CLI arguments (or passed Namespace/list) and updates instance fields."""
        parser = self.create_parser()
        if args is None:
            parsed = parser.parse_args()
        elif isinstance(args, list):
            parsed = parser.parse_args(args)
        elif isinstance(args, argparse.Namespace):
            parsed = args
        else:
            raise TypeError(f"Unsupported args type: {type(args)}")

        self.speaker_list = bool(parsed.speaker_list)
        self.checkpoint_list = bool(parsed.checkpoint_list)
        self.speaker = parsed.speaker
        self.checkpoint = parsed.checkpoint
        self.input_file = parsed.input_file
        self.text = parsed.text
        self.clone_wav = parsed.clone_wav
        self.clone_txt = parsed.clone_txt
        self.output = parsed.output
        # Handle format: List[Format]
        parsed_format = parsed.format
        if parsed_format is not None:
            self.format = Format.normalize(parsed_format)
        elif parsed.output and str(parsed.output).lower().endswith(".mp3"):
            self.format = [Format.MP3]
        else:
            self.format = [Format.WAV]
        self.top_k = int(parsed.top_k)
        self.temperature = float(parsed.temperature)
        self.mbd = bool(parsed.mbd)
        self.nikud_converter = bool(parsed.nikud_converter)

        return parsed

    def run(self, args: Optional[Union[argparse.Namespace, List[str]]] = None) -> Optional[Path]:
        """Main execution workflow for single-sample speech synthesis."""
        if (
            args is not None
            or (self.text is None and self.input_file is None and not self.speaker_list and not self.checkpoint_list)
        ):
            self.parse_args(args)

        speakers = self.speakers.list()

        if self.speaker_list:
            self.print_speakers()
            return None

        if self.checkpoint_list:
            self.print_checkpoints()
            return None

        checkpoint_path = self.resolve_checkpoint(self.checkpoint)
        if not self.clone_wav:
            speaker_id, speaker_data = self.resolve_speaker(self.speaker, speakers)
        else:
            speaker_id, speaker_data = None, None

        # Resolve text input
        lines = None

        input_file_path = None
        if self.input_file is not None:
            input_file_path = Path(self.input_file)
            if not input_file_path.exists():
                print(f"Error: Input file '{self.input_file}' not found.")
                sys.exit(1)
        elif self.text is not None:
            lines = [self.text.strip()]
        elif os.path.exists("samples/sample.txt"):
            input_file_path = Path("samples/sample.txt")

        if input_file_path is not None:
            with open(input_file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.read().splitlines() if line.strip()]

        if not lines:
            lines = ["שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית."]

        # Resolve output path base
        if self.output is not None:
            out_p = Path(self.output)
            if out_p.is_dir() or str(self.output).endswith(("\\", "/")):
                stem = input_file_path.stem if input_file_path is not None else "sample"
                base_out = out_p / stem
            elif out_p.suffix:
                base_out = out_p.with_suffix("")
            else:
                base_out = out_p
        elif input_file_path is not None:
            base_out = input_file_path.with_suffix("")
        else:
            base_out = Path("sample")
        base_out.parent.mkdir(parents=True, exist_ok=True)

        # Resolve prompt audio and text
        if self.clone_wav:
            prompt_audio = self.clone_wav
            if not os.path.exists(prompt_audio):
                print(f"Error: Clone audio file '{prompt_audio}' not found.")
                sys.exit(1)

            if self.clone_txt:
                if os.path.exists(self.clone_txt):
                    with open(self.clone_txt, "r", encoding="utf-8") as f:
                        prompt_text = f.read().strip()
                elif "\n" in self.clone_txt or " " in self.clone_txt:
                    prompt_text = self.clone_txt.strip()
                else:
                    print(f"Error: Clone text file '{self.clone_txt}' not found.")
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
            prompt_audio = speaker_data["audio-prompt"]
            prompt_text = speaker_data["text-prompt"]

        # Device detection
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        print(f"Selected Checkpoint: {checkpoint_path}")
        if self.clone_wav:
            print(f"Clone Audio:         {prompt_audio}")
            print(f"Clone Text:          {prompt_text[:60]}{'...' if len(prompt_text) > 60 else ''}")
        else:
            print(f"Selected Speaker:    {speaker_id}")
        print(f"Device:              {device} ({device_name})")
        if self.input_file:
            print(f"Input File:          {self.input_file}")
        print("Loading model...")

        model, _ = self.load_model(checkpoint_path, device)
        text_collater = get_text_token_collater(self.tokens_file)
        audio_tokenizer = AudioTokenizer(mbd=self.mbd, device=device)
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
            top_k=self.top_k,
            temperature=self.temperature,
            mbd=self.mbd,
            nikud_converter=self.nikud_converter,
        )

        saved_path: Optional[Path] = None
        if Format.WAV in self.format:
            wav_path = base_out.with_suffix(".wav")
            torchaudio.save(str(wav_path), samples, 24000)
            print(f"Saved generated audio (WAV) to: {wav_path}")
            saved_path = wav_path

        if Format.MP3 in self.format:
            mp3_path = base_out.with_suffix(".mp3")
            torchaudio.save(str(mp3_path), samples, 24000)
            print(f"Saved generated audio (MP3) to: {mp3_path}")
            if saved_path is None or (self.output and str(self.output).lower().endswith(".mp3")):
                saved_path = mp3_path

        return saved_path


SampleMake = SampleMaker
Sampler = SampleMaker
SamplerParser = SampleMakerParser


if __name__ == "__main__":
    SampleMaker().run()
