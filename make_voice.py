import argparse

import json
import os
import re
import random

import sys

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple, Union


import numpy as np
import soundfile as sf

import torch

import torchaudio

from tqdm import tqdm


import utils  # Sets up soundfile audio IO fallback and suppresses torchvision warning

from config import (

    Config,

    MbdMode,

    RandomMode,

    NikudMode,

    SpeakerStore,

    audio_to_base64,

    base64_to_audio,
)

from GradioPage import GradioPage

from sample import Sampler

from valle.data import AudioTokenizer, tokenize_audio

from valle.data.collation import get_text_token_collater

from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars

from nikud_converter import convert_nikud_text, has_nikud



def split_text(text: str, max_len: int = 200) -> List[str]:
    """

    Splits Hebrew text into natural clauses and sentences not exceeding max_len characters.

    Preserves line breaks and sentence context to prevent VALL-E attention overflow.
    """

    all_combined_parts: List[str] = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("---") or line.startswith("["):
            continue


        # Clean problematic punctuation / dashes

        line = line.replace("_", " ")

        line = line.replace("—", "-").replace("–", "-")

        line = line.replace("’", "'").replace("‘", "'")


        # 1. Split by major sentence boundaries (.!?)

        sentences = re.split(r'(?<=[.!?])\s+', line)

        parts: List[str] = []


        for s in sentences:

            s = s.strip()

            if not s:
                continue


            if len(s) <= max_len:
                parts.append(s)
                continue


            # 2. Split by secondary clause boundaries (,;:)

            sub_sentences = re.split(r'(?<=[,;:])\s+', s)

            current_part = ""


            for sub in sub_sentences:

                sub = sub.strip()

                if not sub:
                    continue


                if len(sub) > max_len:

                    # 3. If still exceeding max_len, split by words

                    words = sub.split(" ")

                    for w in words:

                        if len(current_part) + len(w) + 1 > max_len:

                            if current_part:
                                parts.append(current_part.strip())

                            current_part = w + " "

                        else:

                            current_part += w + " "

                else:

                    if len(current_part) + len(sub) + 1 > max_len:

                        if current_part:
                            parts.append(current_part.strip())

                        current_part = sub + " "

                    else:

                        current_part += sub + " "


            if current_part:
                parts.append(current_part.strip())


        # Combine small parts up to max_len

        combined_parts: List[str] = []

        current_part = ""

        for p in parts:

            if len(current_part) + len(p) + 1 <= max_len:

                current_part += (" " if current_part else "") + p

            else:

                if current_part:

                    combined_parts.append(current_part)

                current_part = p

        if current_part:

            combined_parts.append(current_part)


        all_combined_parts.extend(combined_parts)


    return all_combined_parts



def format_srt_time(seconds: float) -> str:

    """Formats seconds into standard SRT timestamp: HH:MM:SS,mmm"""

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    secs = int(seconds % 60)

    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"



class VoiceMaker:
    """

    Batch speech generator for .mki script files with dynamic settings overrides.

    Caches model weights and reference audio prompts for optimal GPU throughput.
    """


    def __init__(self, page: Optional[GradioPage] = None, sampler: Optional[Sampler] = None) -> None:

        self.page: GradioPage = page or GradioPage()

        self.sampler: Sampler = sampler or self.page.sampler


        # Cache reference audio embeddings per prompt

        self._cached_prompt_audio: Optional[str] = None

        self._cached_audio_prompts: Optional[torch.Tensor] = None

        self._cached_prompt_text: Optional[str] = None


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

        speaker_mode = config.speaker_mode


        if speaker_mode == "Predefined Speaker":

            spk_data = SpeakerStore()

            spk = spk_data.find(config.predefined_speaker)

            if not spk:

                speakers = spk_data.list()

                spk = speakers[0] if speakers else None

            if not spk:

                raise ValueError("No predefined speakers available.")

            return str(spk.full_path), spk.txt


        # Custom Voice Clone

        clone_audio = config.clone_audio

        clone_text = config.clone_text


        if isinstance(clone_audio, dict):

            # Self-contained Base64 encoded audio

            decoded_path = base64_to_audio(clone_audio)

            if not decoded_path:

                raise ValueError("Failed to decode base64 clone_audio.")

            prompt_audio = decoded_path

        elif isinstance(clone_audio, str) and os.path.exists(clone_audio):

            prompt_audio = clone_audio

        else:

            raise ValueError(f"Invalid clone_audio provided in config: {clone_audio}")


        return prompt_audio, str(clone_text)


    def synthesize_chunk(

        self,

        chunk_text: str,

        config: Config,

        model: torch.nn.Module,

        audio_tokenizer: AudioTokenizer,

        text_collater: Any,

        alef_bert_tokenizer: AlefBERTRootTokenizer,

    ) -> np.ndarray:

        """Synthesizes a single chunk using cached model and prompt representations."""

        prompt_audio, prompt_text = self.resolve_speaker_prompt(config)


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

        return audio_np


    def run(self, args: Optional[Union[argparse.Namespace, List[str]]] = None) -> Path:

        """Main execution workflow for processing .mki files."""

        if args is None:

            args = self.create_parser().parse_args()

        elif isinstance(args, list):

            args = self.create_parser().parse_args(args)


        input_path = Path(args.input).resolve()

        basename = input_path.stem

        output_dir = self.resolve_output_dir(input_path, args.output_dir)


        print(f"\n{'='*70}")

        print(f"🎙️  HebTTS Batch Voice Generator: {input_path.name}"
)
        print(f"Output Directory: {output_dir}")

        print(f"Device:           {self.page.device} ({self.page.device_name})")

        print(f"{'='*70}\n")


        # 1. Load base configuration and read lines

        base_config = self.load_base_config(args.base_config)

        lines = self.read_mki_file(input_path)

        blocks = self.parse_mki(lines, base_config)


        if not blocks:

            print("No valid text lines found in .mki file.")

            sys.exit(0)


        # 2. Deconstruct into manageable text chunks

        all_chunks: List[Tuple[Config, str]] = []

        for block_config, block_text in blocks:

            parts = split_text(block_text, max_len=args.max_len)

            for part in parts:

                all_chunks.append((block_config, part))


        print(f"Found {len(blocks)} configuration blocks -> {len(all_chunks)} text chunks to synthesize.")


        all_wavs: List[np.ndarray] = []

        srt_entries: List[Tuple[float, float, str]] = []

        current_time_sec = 0.0

        sample_rate = 24000


        # 3. Generate audio chunk by chunk

        for chunk_idx, (chunk_config, chunk_text) in enumerate(tqdm(all_chunks, desc="Generating Audio"), start=1):

            checkpoint = chunk_config.checkpoint

            mbd = bool(chunk_config.mbd)


            # Ensure model and tokenizers are loaded in memory

            model, audio_tokenizer, text_collater, alef_bert_tokenizer = self.page.load_model_if_needed(

                checkpoint, mbd
            )


            # Synthesize chunk

            audio_np = self.synthesize_chunk(

                chunk_text=chunk_text,

                config=chunk_config,

                model=model,

                audio_tokenizer=audio_tokenizer,

                text_collater=text_collater,

                alef_bert_tokenizer=alef_bert_tokenizer,
            )


            duration = len(audio_np) / float(sample_rate)

            start_time = current_time_sec

            end_time = current_time_sec + duration

            srt_entries.append((start_time, end_time, chunk_text))


            # Save individual chunk wav and txt

            chunk_wav_path = output_dir / f"{basename}_{chunk_idx:03d}.wav"

            sf.write(str(chunk_wav_path), audio_np, sample_rate)


            chunk_txt_path = output_dir / f"{basename}_{chunk_idx:03d}.txt"

            with open(chunk_txt_path, "w", encoding="utf-8") as f:

                f.write(chunk_text)


            all_wavs.append(audio_np)

            current_time_sec = end_time


            # Inter-sentence pause

            pause_sec = float(chunk_config.pause_duration)

            if chunk_idx < len(all_chunks) and pause_sec > 0:

                silence_len = int(sample_rate * pause_sec)

                silence = np.zeros(silence_len, dtype=audio_np.dtype)

                all_wavs.append(silence)

                current_time_sec += pause_sec


        # 4. Save merged audio and subtitles

        if all_wavs:

            final_wav = np.concatenate(all_wavs, axis=-1)

            final_wav_path = output_dir / f"{basename}.wav"

            sf.write(str(final_wav_path), final_wav, sample_rate)

            print(f"\n[+] Saved merged audio: {final_wav_path} ({current_time_sec:.2f}s)")


            final_srt_path = output_dir / f"{basename}.srt"

            with open(final_srt_path, "w", encoding="utf-8") as f:

                for idx, (st, en, txt) in enumerate(srt_entries, start=1):

                    f.write(f"{idx}\n")

                    f.write(f"{format_srt_time(st)} --> {format_srt_time(en)}\n")

                    f.write(f"{txt}\n\n")

            print(f"[+] Saved subtitles:    {final_srt_path}")


        print(f"\nBatch generation completed successfully! Outputs are in: {output_dir}\n")
        return output_dir


    def create_parser(self) -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser(

            description="HebTTS Batch Voice Generator: Synthesize speech and subtitles from .mki script files"
        )

        parser.add_argument(

            "-i", "--input",

            type=str,

            required=True,

            help="Path to .mki script file (required)",
        )

        parser.add_argument(

            "-c", "--base_config",

            type=str,

            default=None,

            help="Path to base JSON settings file (exported from GradioPage)",
        )

        parser.add_argument(

            "-o", "--output_dir",

            type=str,

            default=None,

            help="Output directory (defaults to <input_dir>/out/)",
        )

        parser.add_argument(

            "--max_len",

            type=int,

            default=200,

            help="Maximum character length per text chunk before splitting (default: 200)",
        )
        return parser



if __name__ == "__main__":

    VoiceMaker().run()
