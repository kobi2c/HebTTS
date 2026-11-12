import argparse
import json
import os
import re
import random
import soundfile as sf
import numpy as np
from tqdm import tqdm

from gradio_interface import SETTINGS_KEYS, base64_to_audio, generate_audio

def split_text(text, max_len=200):
    all_combined_parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith("---") or line.startswith("["): 
            continue
        
        # First split by major sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', line)
        parts = []
        
        for s in sentences:
            s = s.strip()
            if not s: continue
            
            if len(s) <= max_len:
                parts.append(s)
                continue
                
            # If still too long, split by commas, colons, semicolons
            sub_sentences = re.split(r'(?<=[,;:])\s+', s)
            current_part = ''
            
            for sub in sub_sentences:
                sub = sub.strip()
                if not sub: continue
                
                if len(sub) > max_len:
                    # If STILL too long, split by words
                    words = sub.split(' ')
                    for w in words:
                        if len(current_part) + len(w) + 1 > max_len:
                            if current_part:
                                parts.append(current_part.strip())
                            current_part = w + ' '
                        else:
                            current_part += w + ' '
                else:
                    if len(current_part) + len(sub) + 1 > max_len:
                        if current_part:
                            parts.append(current_part.strip())
                        current_part = sub + ' '
                    else:
                        current_part += sub + ' '
                        
            if current_part:
                parts.append(current_part.strip())
                
        # Combine small parts if possible
        combined_parts = []
        current_part = ''
        for p in parts:
            if len(current_part) + len(p) + 1 <= max_len:
                current_part += (' ' if current_part else '') + p
            else:
                if current_part:
                    combined_parts.append(current_part)
                current_part = p
        if current_part:
            combined_parts.append(current_part)
            
        all_combined_parts.extend(combined_parts)
        
    return all_combined_parts

def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def main():
    parser = argparse.ArgumentParser(description="Generate voice from text using exported Gradio settings")
    parser.add_argument("--input"      , "-i" , type=str, required=True, help="Path to text file")
    parser.add_argument("--config"     , "-c" , type=str, required=True, help="Path to JSON settings file")
    parser.add_argument("--output_dir" , "-o" , type=str, default=".", help="Output directory for wav files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(args.input, "r", encoding="cp1252") as f:
            text = f.read()

    # Replace em-dashes and en-dashes with standard hyphens
    text = text.replace("_", " ")
    text = text.replace("—", "-").replace("–", "-")
    # Remove apostrophes which can cause issues with the phonemizer/TTS
    text = text.replace("’", "").replace("‘", "")

    parts = split_text(text, max_len=200)
    print(f"Split text into {len(parts)} parts.")

    with open(args.config, "r", encoding="utf-8") as f:
        settings = json.load(f)

    # prepare base arguments
    base_args = []
    for key in SETTINGS_KEYS:
        val = settings.get(key)
        if key in ["speaker_audio", "prefix_audio"]:
            if val is not None and isinstance(val, dict):
                base_args.append(base64_to_audio(val))
            else:
                base_args.append(None)
        else:
            base_args.append(val)

    # Indexes of varying parameters
    idx_text = SETTINGS_KEYS.index("text")
    idx_pitch = SETTINGS_KEYS.index("pitch_std_slider")
    idx_rate = SETTINGS_KEYS.index("speaking_rate_slider")
    idx_seed = SETTINGS_KEYS.index("seed_number")
    idx_rand_seed = SETTINGS_KEYS.index("randomize_seed_toggle")
    
    orig_pitch = float(base_args[idx_pitch]) if base_args[idx_pitch] is not None else 45.0
    orig_rate = float(base_args[idx_rate]) if base_args[idx_rate] is not None else 15.0

    all_wavs = []
    srt_entries = []
    current_time_sec = 0.0
    global_sr = None

    for i, part in enumerate(tqdm(parts, desc="Generating Audio")):
        call_args = list(base_args)
        call_args[idx_text] = part
        
        # add variance (+/- 5%)
        call_args[idx_pitch] = orig_pitch * random.uniform(0.95, 1.05)
        call_args[idx_rate] = orig_rate * random.uniform(0.95, 1.05)
        
        # force random seed for variance
        call_args[idx_seed] = random.randint(0, 2**31 - 1)
        call_args[idx_rand_seed] = False
        
        # generate
        # generate_audio returns ((sr_out, wav_out), seed)
        res, _ = generate_audio(*call_args, progress=lambda x: None)
        sr_out, wav_out = res
        
        all_wavs.append(wav_out)
        duration = wav_out.shape[-1] / sr_out
        srt_entries.append((current_time_sec, current_time_sec + duration, part))
        current_time_sec += duration
        global_sr = sr_out
        
        # save
        basename = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(args.output_dir, f"{basename}_{i+1:03d}.wav")
        sf.write(out_path, wav_out, sr_out)
        
        txt_out_path = os.path.join(args.output_dir, f"{basename}_{i+1:03d}.txt")
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write(part)
            
    if all_wavs:
        # Concatenate on the last axis (time axis)
        final_wav = np.concatenate(all_wavs, axis=-1)
        final_out_path = os.path.join(args.output_dir, f"{basename}.wav")
        sf.write(final_out_path, final_wav, global_sr)
        print(f"Successfully saved merged audio to {final_out_path}")
        
        final_srt_path = os.path.join(args.output_dir, f"{basename}.srt")
        with open(final_srt_path, "w", encoding="utf-8") as f:
            for idx, (start, end, text_part) in enumerate(srt_entries):
                f.write(f"{idx + 1}\n")
                f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
                f.write(f"{text_part}\n\n")
        print(f"Successfully saved subtitles to {final_srt_path}")
        
if __name__ == "__main__":
    main()
