#!/usr/bin/env python3
"""
Script to extract and decode embedded files (like base64 audio) from config JSON files.
Extracted files are named: <config-name>-<key>.<ext>
Decoding logic follows gradio_interface.py (base64_to_audio).
"""

import argparse
import base64
import glob
import json
import os
import sys
from pathlib import Path


def decode_embedded_data(b64_dict):
    """
    Decodes an embedded file dictionary containing base64 data.
    Matches the decoding logic in gradio_interface.py.
    """
    if not isinstance(b64_dict, dict) or "data" not in b64_dict:
        return None, None

    data_str = b64_dict.get("data")
    if not data_str:
        return None, None

    ext = b64_dict.get("ext", ".wav")
    if not ext.startswith("."):
        ext = f".{ext}"

    try:
        decoded_bytes = base64.b64decode(data_str)
        return decoded_bytes, ext
    except Exception as e:
        print(f"  [Error] Failed to decode base64 data: {e}", file=sys.stderr)
        return None, None


def extract_from_config(config_path, output_dir=None):
    """
    Reads a config file and extracts all embedded files.
    Names output files as <config-name>-<key>.<ext>.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        print(f"[Warning] Config file not found: {config_path}", file=sys.stderr)
        return []

    config_name = config_file.stem
    target_dir = Path(output_dir) if output_dir else config_file.parent

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            settings = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to read JSON from '{config_path}': {e}", file=sys.stderr)
        return []

    if not isinstance(settings, dict):
        print(f"[Warning] JSON content in '{config_path}' is not an object/dictionary.", file=sys.stderr)
        return []

    extracted_files = []
    print(f"\nProcessing '{config_file.name}'...")

    for key, value in settings.items():
        if isinstance(value, dict) and "data" in value:
            decoded_bytes, ext = decode_embedded_data(value)
            if decoded_bytes is None:
                continue

            output_filename = f"{config_name}-{key}{ext}"
            output_filepath = target_dir / output_filename

            try:
                with open(output_filepath, "wb") as out_f:
                    out_f.write(decoded_bytes)
                print(f"  -> Extracted '{key}' ({len(decoded_bytes):,} bytes) to: {output_filepath}")
                extracted_files.append(output_filepath)
            except Exception as e:
                print(f"  [Error] Failed writing '{output_filepath}': {e}", file=sys.stderr)

    if not extracted_files:
        print(f"  No embedded files found in '{config_file.name}'.")

    return extracted_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract embedded base64 files from config JSON files into <config-name>-<key>.<ext>."
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="Path(s) or glob pattern(s) to config JSON file(s). Defaults to 'config*.json' if omitted.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to save extracted files. Defaults to the same directory as each config file.",
    )

    args = parser.parse_args()

    # Resolve config files
    config_paths = []
    if args.configs:
        for pattern in args.configs:
            matches = glob.glob(pattern)
            if matches:
                config_paths.extend(matches)
            else:
                config_paths.append(pattern)
    else:
        # Default: search for config*.json in current working directory
        config_paths = glob.glob("config*.json")
        if not config_paths:
            # Fallback: check all .json in current directory
            config_paths = glob.glob("*.json")

    if not config_paths:
        print("No config JSON files found to process.")
        sys.exit(1)

    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for p in config_paths:
        norm = os.path.normpath(p)
        if norm not in seen:
            seen.add(norm)
            unique_configs.append(p)

    total_extracted = 0
    for cfg in unique_configs:
        extracted = extract_from_config(cfg, args.output_dir)
        total_extracted += len(extracted)

    print(f"\nDone! Total extracted files: {total_extracted}")


if __name__ == "__main__":
    main()
