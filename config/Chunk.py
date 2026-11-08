from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import soundfile as sf

from .Config import Config, ChainMode
from .Format import Format
from .NoiseFix import NoiseFix
from .Range import RangeList


def format_srt_time(seconds: float) -> str:
    """Formats seconds into standard SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def split_text(text: str, max_len: int = 100) -> List[str]:
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


class ChunkStatus(IntEnum):
    """Lifecycle status of an individual audio chunk during batch processing."""
    PENDING = 0
    SYNTHESIZED = 1
    REUSED = 2
    CLEANED = 3
    SKIPPED = 4
    NOT_FOUND = 5

    def __str__(self) -> str:
        return self.name

    @classmethod
    def _missing_(cls, value: Any) -> "ChunkStatus":
        if isinstance(value, str):
            val_upper = value.strip().upper()
            for member in cls:
                if member.name == val_upper:
                    return member
        elif isinstance(value, int):
            for member in cls:
                if member.value == value:
                    return member
        return super()._missing_(value)


@dataclass
class Chunk:
    """Represents a single segment in batch speech synthesis."""
    index: int
    config: Config
    text: str
    audio: Optional[np.ndarray] = None
    sample_rate: int = 24000
    status: ChunkStatus = ChunkStatus.PENDING
    duration: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    tokens_str: str = "-"
    error: Optional[str] = None

    def wav_path(self, output_dir: Union[str, Path], basename: str) -> Path:
        """Returns the destination .wav path for this chunk."""
        return Path(output_dir) / f"{basename}_{self.index:03d}.wav"

    def txt_path(self, output_dir: Union[str, Path], basename: str) -> Path:
        """Returns the destination .txt path for this chunk."""
        return Path(output_dir) / f"{basename}_{self.index:03d}.txt"

    def exists(self, output_dir: Union[str, Path], basename: str) -> bool:
        """Returns True if the chunk wav audio file exists on disk."""
        return self.wav_path(output_dir, basename).exists()

    def load_audio(self, output_dir: Union[str, Path], basename: str) -> np.ndarray:
        """Loads existing audio from disk, updating status to REUSED and calculating duration."""
        wav_file = self.wav_path(output_dir, basename)
        audio_np, sr = sf.read(str(wav_file), dtype="float32")
        self.audio = audio_np
        self.sample_rate = sr
        self.duration = len(audio_np) / float(sr)
        self.status = ChunkStatus.REUSED
        self.tokens_str = "-"
        return audio_np

    def save_audio(self, output_dir: Union[str, Path], basename: str) -> Path:
        """Writes current chunk audio to disk as a 24kHz float32 WAV file."""
        if self.audio is None:
            raise ValueError(f"Chunk {self.index} has no audio to save.")
        wav_file = self.wav_path(output_dir, basename)
        sf.write(str(wav_file), self.audio, self.sample_rate)
        return wav_file

    def save_text(self, output_dir: Union[str, Path], basename: str) -> Path:
        """Writes chunk text to disk as a .txt file."""
        txt_file = self.txt_path(output_dir, basename)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(self.text)
        return txt_file

    def clean(
        self,
        output_dir: Union[str, Path],
        basename: str,
        trim_noise: bool = True,
        pad_ms: float = 30.0,
        fade_ms: float = 20.0,
        apply_fix: bool = True,
    ) -> bool:
        """
        Cleans existing chunk audio on disk using NoiseFix (trim leading noise and apply micro-fade).
        """
        wav_file = self.wav_path(output_dir, basename)
        if not wav_file.exists():
            self.status = ChunkStatus.NOT_FOUND
            return False

        audio_np, sr_disk = sf.read(str(wav_file), dtype="float32")
        if apply_fix:
            if trim_noise:
                audio_np = NoiseFix.trim(audio_np, sr=sr_disk, pad_ms=pad_ms)
            if fade_ms > 0:
                audio_np = NoiseFix.fade(audio_np, sr=sr_disk, fade_in_ms=fade_ms, fade_out_ms=fade_ms)
            sf.write(str(wav_file), audio_np, sr_disk)

        self.audio = audio_np
        self.sample_rate = sr_disk
        self.duration = len(audio_np) / float(sr_disk)
        self.status = ChunkStatus.CLEANED
        self.tokens_str = "-"
        return True

    def synthesize(
        self,
        voice_maker: Any,
        output_dir: Union[str, Path],
        basename: str,
        prompt_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        fade_ms: float = 20.0,
        trim_noise: bool = True,
        pad_ms: float = 30.0,
    ) -> np.ndarray:
        """
        Synthesizes chunk speech via voice_maker, saves chunk wav & txt files,
        and records token usage and status.
        """
        checkpoint = self.config.checkpoint
        mbd = bool(self.config.mbd)

        model, audio_tokenizer, text_collater, alef_bert_tokenizer = voice_maker.page.load_model_if_needed(
            checkpoint, mbd
        )
        model.verbose = False
        model.last_eos = None

        audio_np = voice_maker.synthesize_chunk(
            chunk_text=self.text,
            config=self.config,
            model=model,
            audio_tokenizer=audio_tokenizer,
            text_collater=text_collater,
            alef_bert_tokenizer=alef_bert_tokenizer,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            fade_ms=fade_ms,
            trim_noise=trim_noise,
            pad_ms=pad_ms,
        )

        self.audio = audio_np
        self.sample_rate = 24000
        self.duration = len(audio_np) / float(self.sample_rate)

        # Save individual chunk wav and txt
        self.save_audio(output_dir, basename)
        self.save_text(output_dir, basename)

        self.status = ChunkStatus.SYNTHESIZED
        eos = model.last_eos
        if eos and isinstance(eos, (tuple, list)) and len(eos) >= 2:
            p_len, tot_len = eos[0], eos[1]
            diff = tot_len - p_len
            self.tokens_str = f"{p_len} -> {tot_len} (+{diff})"
        else:
            self.tokens_str = "-"
        return audio_np

    def format_table_row(self, total_chunks: int) -> str:
        """Formats single row string for the CLI progress table."""
        text_preview = self.text[:35] + "..." if len(self.text) > 35 else self.text
        duration_str = f"{self.duration:.2f}s" if self.duration > 0 else "-"
        return f"[{self.index:>2}/{total_chunks:<2}]    {self.status.name:<14} {duration_str:<10} {self.tokens_str:<24} {text_preview}"

    def to_srt_block(self, srt_index: int) -> str:
        """Formats the chunk into an SRT subtitle cue block."""
        return (
            f"{srt_index}\n"
            f"{format_srt_time(self.start_time)} --> {format_srt_time(self.end_time)}\n"
            f"{self.text}\n\n"
        )


class ChunkList:
    """Manages an ordered collection of Chunk instances for batch processing and export."""

    def __init__(self, chunks: Optional[Sequence[Chunk]] = None, sample_rate: int = 24000) -> None:
        self.chunks: List[Chunk] = list(chunks) if chunks is not None else []
        self.sample_rate: int = sample_rate
        self.target_indices: Optional[Set[int]] = None
        self.last_target_idx: int = -1

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Chunk, List[Chunk]]:
        if isinstance(idx, slice):
            return self.chunks[idx]
        return self.chunks[idx]

    def append(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    split_text = staticmethod(split_text)

    @classmethod
    def from_blocks(cls, blocks: List[Tuple[Config, str]], max_len: int = 100) -> "ChunkList":
        """Constructs a ChunkList from parsed MKI blocks [(Config, text)]."""
        chunk_items: List[Chunk] = []
        idx = 1
        for block_config, block_text in blocks:
            parts = split_text(block_text, max_len=max_len)
            for part in parts:
                chunk_items.append(Chunk(index=idx, config=block_config, text=part))
                idx += 1
        return cls(chunk_items)

    def filter_targets(
        self,
        active_range: Optional[RangeList] = None,
        range_only: bool = False,
    ) -> Tuple[Set[int], int]:
        """Calculates set of target chunk indices to process/join, and the last target index."""
        total = len(self.chunks)
        if range_only and active_range is not None:
            target_indices = set(active_range.indices(total))
        else:
            target_indices = set(range(1, total + 1))
        last_target_idx = max(target_indices) if target_indices else -1
        self.target_indices = target_indices
        self.last_target_idx = last_target_idx
        return target_indices, last_target_idx

    def resolve_prompt(
        self,
        chunk_idx: int,
        output_dir: Union[str, Path],
        basename: str,
        chain_speaker: ChainMode = ChainMode.Enabled,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolves prompt audio path and transcript for chunk_idx (0-based list index).
        If chain_speaker is enabled and previous chunk shares the same speaker identity
        and its audio/text exist, returns (prev_wav_path, prev_text).
        Otherwise returns (None, None), signalling to use the base speaker.
        """
        if not bool(chain_speaker) or chunk_idx <= 0 or chunk_idx >= len(self.chunks):
            return None, None

        prev_chunk = self.chunks[chunk_idx - 1]
        curr_chunk = self.chunks[chunk_idx]

        # Verify speaker identity did not change via ! or @ directives
        prev_spk = (
            prev_chunk.config.speaker_mode,
            prev_chunk.config.predefined_speaker,
            prev_chunk.config.clone_audio,
        )
        curr_spk = (
            curr_chunk.config.speaker_mode,
            curr_chunk.config.predefined_speaker,
            curr_chunk.config.clone_audio,
        )
        if prev_spk != curr_spk:
            return None, None

        out_path = Path(output_dir)
        prev_wav = prev_chunk.wav_path(out_path, basename)
        prev_txt = prev_chunk.txt_path(out_path, basename)

        # Check if previous chunk audio and text exist on disk
        if prev_wav.exists() and prev_txt.exists():
            try:
                txt_content = prev_txt.read_text(encoding="utf-8").strip()
            except Exception:
                txt_content = prev_chunk.text
            if txt_content:
                return str(prev_wav), txt_content

        # Fallback to in-memory previous chunk if available and saved
        if prev_chunk.audio is not None and prev_chunk.text:
            if prev_wav.exists():
                return str(prev_wav), prev_chunk.text

        return None, None

    def process(
        self,
        voice_maker: Any,
        output_dir: Union[str, Path],
        basename: str,
        active_range: Optional[RangeList] = None,
        range_only: bool = False,
        chain_speaker: Optional[ChainMode] = None,
        clean_existing: bool = False,
        trim_noise: bool = True,
        pad_ms: float = 30.0,
        fade_ms: float = 20.0,
    ) -> Dict[str, int]:
        """
        Executes batch processing across all chunks with real-time table progress reporting.
        """
        out_dir = Path(output_dir)
        total_chunks = len(self.chunks)
        target_indices, last_target_idx = self.filter_targets(active_range, range_only)

        synthesized_count = 0
        reused_count = 0
        skipped_count = 0
        cleaned_count = 0
        batch_start_time = time.time()
        current_time_sec = 0.0

        print("\n" + "=" * 95)
        print(f"{'Chunk':<10} {'Status':<14} {'Audio':<10} {'Tokens (Prompt->Total)':<24} Text")
        print("-" * 95)

        for chunk_idx, chunk in enumerate(self.chunks):
            if range_only and chunk.index not in target_indices:
                chunk.status = ChunkStatus.SKIPPED
                skipped_count += 1
                print(chunk.format_table_row(total_chunks))
                continue

            if clean_existing:
                if chunk.exists(out_dir, basename):
                    apply_fix = (active_range is None or chunk.index in active_range)
                    chunk.clean(
                        out_dir,
                        basename,
                        trim_noise=trim_noise,
                        pad_ms=pad_ms,
                        fade_ms=fade_ms,
                        apply_fix=apply_fix,
                    )
                    cleaned_count += 1
                else:
                    chunk.status = ChunkStatus.NOT_FOUND
                    skipped_count += 1
                    print(chunk.format_table_row(total_chunks))
                    continue
            else:
                should_synthesize = True
                if active_range is not None and chunk.index not in active_range:
                    if chunk.exists(out_dir, basename):
                        should_synthesize = False

                if should_synthesize:
                    # Resolve prompt audio and text (chaining from previous chunk if enabled)
                    effective_chain = chain_speaker if chain_speaker is not None else chunk.config.chain_speaker

                    prompt_audio, prompt_text = self.resolve_prompt(
                        chunk_idx=chunk_idx,
                        output_dir=out_dir,
                        basename=basename,
                        chain_speaker=effective_chain,
                    )
                    chunk.synthesize(
                        voice_maker=voice_maker,
                        output_dir=out_dir,
                        basename=basename,
                        prompt_audio=prompt_audio,
                        prompt_text=prompt_text,
                        fade_ms=fade_ms,
                        trim_noise=trim_noise,
                        pad_ms=pad_ms,
                    )
                    synthesized_count += 1
                else:
                    chunk.load_audio(out_dir, basename)
                    reused_count += 1

            print(chunk.format_table_row(total_chunks))

            chunk.start_time = current_time_sec
            chunk.end_time = current_time_sec + chunk.duration
            current_time_sec = chunk.end_time

            # Inter-sentence pause
            pause_sec = float(chunk.config.pause_duration)
            if chunk.index != last_target_idx and pause_sec > 0:
                current_time_sec += pause_sec

        total_elapsed = time.time() - batch_start_time
        print("-" * 95)
        if clean_existing:
            print(f"Summary: {cleaned_count} cleaned, {skipped_count} skipped ({total_chunks} total chunks) in {total_elapsed:.1f}s")
        else:
            print(f"Summary: {synthesized_count} synthesized, {reused_count} reused, {skipped_count} skipped ({total_chunks} total chunks) in {total_elapsed:.1f}s")
        print("=" * 95)

        return {
            "synthesized": synthesized_count,
            "reused": reused_count,
            "skipped": skipped_count,
            "cleaned": cleaned_count,
            "total": total_chunks,
        }

    def join_audio(
        self,
        target_indices: Optional[Set[int]] = None,
        last_target_idx: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Concatenates chunk audio with inter-sentence pauses into a single continuous waveform.
        Returns (merged_wav_numpy, total_duration_seconds).
        """
        if target_indices is None:
            target_indices = self.target_indices
        if last_target_idx is None:
            last_target_idx = self.last_target_idx

        active_chunks = [
            c for c in self.chunks
            if c.audio is not None and (target_indices is None or c.index in target_indices)
        ]
        if not active_chunks:
            return None, 0.0

        if last_target_idx is None or last_target_idx < 0:
            last_target_idx = active_chunks[-1].index

        all_wavs: List[np.ndarray] = []
        for chunk in active_chunks:
            all_wavs.append(chunk.audio)
            pause_sec = float(chunk.config.pause_duration)
            if chunk.index != last_target_idx and pause_sec > 0:
                silence_len = int(round(self.sample_rate * pause_sec))
                silence = np.zeros(silence_len, dtype=chunk.audio.dtype)
                all_wavs.append(silence)

        merged = np.concatenate(all_wavs, axis=-1)
        total_duration = len(merged) / float(self.sample_rate)
        return merged, total_duration

    def save_merged(
        self,
        output_dir: Union[str, Path],
        basename: str,
        formats: Union[Format, List[Format]],
        merged_wav: np.ndarray,
        sample_rate: int = 24000,
    ) -> List[Path]:
        """Writes merged audio to disk in the specified audio formats (WAV and/or MP3)."""
        norm_formats = Format.normalize(formats)
        saved_paths: List[Path] = []
        out_dir = Path(output_dir)
        total_duration = len(merged_wav) / float(sample_rate)

        if Format.WAV in norm_formats:
            final_wav_path = out_dir / f"{basename}.wav"
            sf.write(str(final_wav_path), merged_wav, sample_rate)
            print(f"\n[+] Saved merged audio (WAV): {final_wav_path} ({total_duration:.2f}s)")
            saved_paths.append(final_wav_path)

        if Format.MP3 in norm_formats:
            final_mp3_path = out_dir / f"{basename}.mp3"
            sf.write(str(final_mp3_path), merged_wav, sample_rate, format="MP3")
            print(f"\n[+] Saved merged audio (MP3): {final_mp3_path} ({total_duration:.2f}s)")
            saved_paths.append(final_mp3_path)

        return saved_paths

    def save_srt(
        self,
        output_dir: Union[str, Path],
        basename: str,
        target_indices: Optional[Set[int]] = None,
    ) -> Path:
        """Writes subtitle (.srt) file for all processed target chunks."""
        if target_indices is None:
            target_indices = self.target_indices

        out_dir = Path(output_dir)
        final_srt_path = out_dir / f"{basename}.srt"
        active_chunks = [
            c for c in self.chunks
            if c.audio is not None and (target_indices is None or c.index in target_indices)
        ]

        with open(final_srt_path, "w", encoding="utf-8") as f:
            for srt_idx, chunk in enumerate(active_chunks, start=1):
                f.write(chunk.to_srt_block(srt_idx))

        print(f"[+] Saved subtitles:          {final_srt_path}")
        return final_srt_path
