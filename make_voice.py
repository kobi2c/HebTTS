import argparse
import json
import os
import re
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import utils
import numpy as np
import soundfile as sf
import torch
import torchaudio
from config import (
    Config,
    SpeakerMode,
    Format,
    MbdMode,
    RandomMode,
    NikudMode,
    ChainMode,
    SpeakerStore,
    Range,
    RangeList,
    NoiseFix,
    Chunk,
    ChunkList,
    ChunkStatus,
    split_text,
    format_srt_time,
)
from GradioPage import GradioPage
from make_sample import SampleMaker
from valle.data import AudioTokenizer, tokenize_audio
from valle.data.collation import get_text_token_collater
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars
from nikud_converter import convert_nikud_text, has_nikud


class VoiceMakerParser(argparse.ArgumentParser):
    """Command-line argument parser for HebTTS Batch Voice Generator."""

    def __init__(self, **kwargs):
        super().__init__(
            description="HebTTS Batch Voice Generator: Synthesize speech and subtitles from .mki script files",
            **kwargs,
        )
        self.add_argument(
            "-i", "--input",
            type=str,
            required=True,
            help="Path to .mki script file (required)",
        )
        self.add_argument(
            "-c", "--base_config",
            type=str,
            default=None,
            help="Path to base JSON settings file (exported from GradioPage)",
        )
        self.add_argument(
            "-o", "--output_dir",
            type=str,
            default=None,
            help="Output directory (defaults to <input_dir>/out/)",
        )
        self.add_argument(
            "--max_chunk_size",
            dest="max_chunk_size",
            type=int,
            default=None,
            help="Maximum character length per text chunk before splitting (defaults to base config max_chunk_size)",
        )
        self.add_argument(
            "--fade_ms",
            type=float,
            default=20.0,
            help="Micro fade-in and fade-out duration in milliseconds to prevent boundary clicks (default: 20.0)",
        )
        self.add_argument(
            "--trim_noise",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Trim leading vocoder startup noise packet before speech onset (default: True, use --no-trim_noise to disable)",
        )
        self.add_argument(
            "--pad_ms",
            type=float,
            default=30.0,
            help="Pre-speech safety silence padding in milliseconds to keep before speech onset (default: 30.0)",
        )
        self.add_argument(
            "--clean_existing",
            action="store_true",
            default=False,
            help="Clean and re-save existing chunk WAV files on disk using noise trimming and fading without re-synthesizing, then rebuild merged WAV and SRT",
        )
        self.add_argument(
            "--range",
            type=str,
            default=None,
            help="Generate only specified chunk index ranges (e.g. '1-5', '@-10', '5-@', 'x-10', '5-x', '3,7-9'). Outside chunks are re-used from disk.",
        )
        self.add_argument(
            "--range-only", "--range_only",
            dest="range_only",
            nargs="?",
            const=True,
            default=False,
            help="Generate and join only the specified range chunks without reading or joining outside chunks from disk. Can be used as a flag (with --range) or passed a range directly (e.g. --range-only '3').",
        )
        self.add_argument(
            "--format",
            type=str,
            choices=["wav", "mp3", "both"],
            default="wav",
            help="Output audio format for merged audio: 'wav', 'mp3', or 'both' (default: 'wav')",
        )


class VoiceMaker:
    """
    Batch speech generator for .mki script files with dynamic settings overrides.
    Caches model weights and reference audio prompts for optimal GPU throughput.
    """
    def __init__(
        self,
        page: Optional[GradioPage] = None,
        sampler: Optional[SampleMaker] = None,
        input: Optional[Union[str, Path]] = None,
        base_config: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        max_chunk_size: Optional[int] = None,
        fade_ms: float = 20.0,
        trim_noise: bool = True,
        pad_ms: float = 30.0,
        clean_existing: bool = False,
        range: Optional[str] = None,
        range_only: bool = False,
        format: Optional[List[Format]] = None,
        chain_speaker: Optional[ChainMode] = None,
    ) -> None:
        self.page: GradioPage = page or GradioPage()
        self.sampler: SampleMaker = sampler or self.page.sampler
        self.input: Optional[Union[str, Path]] = input
        self.base_config: Optional[str] = base_config
        self.output_dir: Optional[Union[str, Path]] = output_dir
        self.max_chunk_size: Optional[int] = int(max_chunk_size) if max_chunk_size is not None else None
        self.fade_ms: float = fade_ms
        self.trim_noise: bool = trim_noise
        self.pad_ms: float = pad_ms
        self.clean_existing: bool = clean_existing
        self.range: Optional[str] = range
        self.range_only: bool = bool(range_only)
        self.format: List[Format] = Format.normalize(format)
        self.chain_speaker: Optional[ChainMode] = ChainMode(int(chain_speaker)) if chain_speaker is not None else None
        # Cache reference audio embeddings per prompt
        self._cached_prompt_audio: Optional[str] = None
        self._cached_audio_prompts: Optional[torch.Tensor] = None
        self._cached_prompt_text: Optional[str] = None

    normalize_formats = Format.normalize

    def create_parser(self) -> VoiceMakerParser:
        """Creates and returns the command line argument parser for VoiceMaker."""
        return VoiceMakerParser()

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

        self.input = parsed.input
        self.base_config = parsed.base_config
        self.output_dir = parsed.output_dir
        self.max_chunk_size = parsed.max_chunk_size
        self.fade_ms = parsed.fade_ms
        self.trim_noise = parsed.trim_noise
        self.pad_ms = parsed.pad_ms
        self.clean_existing = parsed.clean_existing
        self.range = parsed.range

        # Handle --range-only: can be bool flag or range string
        range_only_val = parsed.range_only
        if isinstance(range_only_val, str):
            if not self.range:
                self.range = range_only_val
            self.range_only = True
        else:
            self.range_only = bool(range_only_val)

        # Handle format: List[Format]
        self.format = Format.normalize(parsed.format)

        return parsed


    def resolve_output_dir(self, input_path: Union[str, Path], custom_out: Optional[str] = None) -> Path:
        """Defaults output directory to <input_dir>/out/ if not explicitly given."""
        in_p = Path(input_path).resolve()
        if custom_out:
            out_p = Path(custom_out).resolve()
        else:
            out_p = in_p.parent / "out"
        out_p.mkdir(parents=True, exist_ok=True)
        return out_p
    def load_base_config(self, base_config_path: Optional[str]) -> Config:
        """Loads baseline settings from a JSON file (exported from GradioPage) or default settings."""
        defaults = Config().reset()
        if not base_config_path:
            return defaults
        path = Path(base_config_path)
        if not path.exists():
            print(f"Warning: Base config file '{base_config_path}' not found. Using defaults.")
            return defaults
        try:
            return Config.loadJson(path)
        except Exception as e:
            print(f"Warning: Could not read base config '{base_config_path}': {e}. Using defaults.")
            return defaults
    def read_mki_file(self, filepath: Union[str, Path]) -> List[str]:
        """Reads .mki file safely supporting UTF-8 (with or without BOM) and CP1252."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                return f.readlines()
        except UnicodeDecodeError:
            with open(p, "r", encoding="cp1252") as f:
                return f.readlines()
    def parse_mki(
        self, lines: List[str], base_config: Config
    ) -> List[Tuple[Config, str]]:
        """
        Parses .mki lines grouping text into blocks controlled by dynamic settings directives.
        - Lines starting with '[' are comments.
        - Lines starting with '@' update active settings incrementally.
        - Lines starting with '!' reset to base_config and apply specified updates.
        """
        blocks: List[Tuple[Config, str]] = []
        current_config = Config.from_dict(base_config.to_dict())
        current_block_lines: List[str] = []
        for line_idx, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("["):
                continue
            if line.startswith("@") or line.startswith("!"):
                if current_block_lines:
                    blocks.append((Config.from_dict(current_config.to_dict()), "\n".join(current_block_lines)))
                    current_block_lines = []
                json_str = line[1:].strip()
                try:
                    new_settings = json.loads(json_str)
                    if line.startswith("!"):
                        current_config = Config.from_dict(base_config.to_dict())
                    current_config.from_dict(new_settings)
                except json.JSONDecodeError as e:
                    print(f"Warning [line {line_idx}]: Failed to parse JSON config on line: {line}\nError: {e}")
                continue
            current_block_lines.append(line)
        if current_block_lines:
            blocks.append((Config.from_dict(current_config.to_dict()), "\n".join(current_block_lines)))
        return blocks
    def resolve_speaker_prompt(self, config: Config) -> Tuple[str, str]:
        """Resolves prompt audio path and reference transcript from config."""
        spk = config.speaker
        if not spk:
            raise ValueError("No speaker configured.")
        if spk.mode == SpeakerMode.Clone:
            if not spk.exists and not spk.data:
                raise ValueError(f"No clone audio available for speaker '{spk.id}'.")
            if not spk.text or not spk.text.strip():
                raise ValueError("No clone transcript text configured in speaker.")
        return spk.resolve_audio_path(), spk.text
    def synthesize_chunk(
        self,
        chunk_text: str,
        config: Config,
        model: torch.nn.Module,
        audio_tokenizer: AudioTokenizer,
        text_collater: Any,
        alef_bert_tokenizer: AlefBERTRootTokenizer,
        prompt_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        fade_ms: float = 20.0,
        trim_noise: bool = True,
        pad_ms: float = 30.0,
    ) -> np.ndarray:
        """Synthesizes a single chunk using cached model and prompt representations."""
        if prompt_audio is None or prompt_text is None:
            resolved_audio, resolved_text = self.resolve_speaker_prompt(config)
            if prompt_audio is None:
                prompt_audio = resolved_audio
            if prompt_text is None:
                prompt_text = resolved_text

        # Update cached prompt representations if speaker or audio changed
        if self._cached_prompt_audio != prompt_audio or self._cached_audio_prompts is None:
            encoded_frames = tokenize_audio(audio_tokenizer, prompt_audio)
            self._cached_audio_prompts = torch.concat([encoded_frames[0][0]], dim=-1).transpose(2, 1).to(self.page.device)
            self._cached_prompt_audio = prompt_audio
            self._cached_prompt_text = prompt_text
        # Randomize seed if enabled
        if config.randomize_seed == RandomMode.Enabled:
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = int(config.seed_number)
        torch.manual_seed(seed)
        top_k = int(config.top_k)
        temperature = float(config.temperature)
        mbd = bool(config.mbd)
        nikud_converter = bool(config.nikud_converter)
        seg = self.sampler.synthesize_line(
            text=chunk_text,
            model=model,
            text_collater=text_collater,
            audio_tokenizer=audio_tokenizer,
            alef_bert_tokenizer=alef_bert_tokenizer,
            audio_prompts=self._cached_audio_prompts,
            prompt_text=self._cached_prompt_text or prompt_text,
            device=self.page.device,
            top_k=top_k,
            temperature=temperature,
            mbd=mbd,
            nikud_converter=nikud_converter,
        )
        audio_np = seg.squeeze().cpu().numpy()
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val
        # Trim leading vocoder startup noise and clicks before speech onset
        if trim_noise:
            audio_np = NoiseFix.trim(audio_np, sr=24000, pad_ms=pad_ms)
        # Apply smooth micro fade-in and fade-out to suppress vocoder boundary clicks/transients
        if fade_ms > 0:
            audio_np = NoiseFix.fade(audio_np, sr=24000, fade_in_ms=fade_ms, fade_out_ms=fade_ms)
        return audio_np
    def run(self, args: Optional[Union[argparse.Namespace, List[str]]] = None) -> Path:
        """Main execution workflow for processing .mki files."""
        if args is not None or self.input is None:
            self.parse_args(args)

        if not self.input:
            raise ValueError("Input .mki file path must be specified.")

        input_path = Path(self.input).resolve()
        basename = input_path.stem
        output_dir = self.resolve_output_dir(input_path, self.output_dir)
        print(f"\n{'='*70}")
        print(f"🎙️  HebTTS Batch Voice Generator: {input_path.name}")
        print(f"Output Directory: {output_dir}")
        print(f"Device:           {self.page.device} ({self.page.device_name})")
        print(f"{'='*70}\n")

        # 1. Load base configuration and read lines
        base_config = self.load_base_config(self.base_config)
        lines = self.read_mki_file(input_path)
        blocks = self.parse_mki(lines, base_config)
        if not blocks:
            print("No valid text lines found in .mki file.")
            sys.exit(0)

        # 2. Deconstruct into manageable text chunks
        effective_max_chunk = (
            self.max_chunk_size
            if self.max_chunk_size is not None
            else base_config.max_chunk_size
        )
        chunks = ChunkList.from_blocks(blocks, max_chunk_size=effective_max_chunk)
        print(f"Found {len(blocks)} configuration blocks -> {len(chunks)} text chunks to synthesize.")

        # Parse optional range filter
        active_range: Optional[RangeList] = None
        if self.range:
            active_range = RangeList.parse(self.range).normalize(last=len(chunks))
            indices = active_range.indices(len(chunks))
            label = "Range-Only" if self.range_only else "Range"
            print(f"[{label}] Filter active: {active_range} ({len(indices)} of {len(chunks)} chunks selected).")
        elif self.range_only:
            print("[Range-Only] Note: No range specified, processing all chunks.")

        # 3. Pre-load initial model and MBD vocoder if synthesis is needed
        if not self.clean_existing:
            target_indices, _ = chunks.filter_targets(active_range, self.range_only)
            target_chunk = next((c for c in chunks if c.index in target_indices), None)
            if target_chunk:
                self.page.load_model_if_needed(target_chunk.config.checkpoint, target_chunk.config.mbd)

        # 4. Process or generate audio chunk by chunk
        chunks.process(
            voice_maker=self,
            output_dir=output_dir,
            basename=basename,
            active_range=active_range,
            range_only=self.range_only,
            chain_speaker=self.chain_speaker,
            clean_existing=bool(self.clean_existing),
            trim_noise=bool(self.trim_noise),
            pad_ms=float(self.pad_ms),
            fade_ms=float(self.fade_ms),
        )

        # 5. Save merged audio and subtitles
        merged_wav, total_sec = chunks.join_audio()
        if merged_wav is not None:
            chunks.save_merged(output_dir, basename, self.format, merged_wav, sample_rate=24000)
            chunks.save_srt(output_dir, basename)

        print(f"\nBatch generation completed successfully! Outputs are in: {output_dir}\n")
        return output_dir


if __name__ == "__main__":
    VoiceMaker().run()
