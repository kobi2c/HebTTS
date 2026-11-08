#!/usr/bin/env python3
"""
speaker_same.py - Standalone CLI for biometric speaker verification.
Evaluates whether a test audio file is from the same speaker as a reference audio file,
returning a calibrated 0-100 certainty score.

Usage:
    python speaker_same.py speakers/osim.wav data/out/a1_001.wav
    python speaker_same.py speakers/osim.wav data/out/a1_003.wav --threshold 90
    python speaker_same.py speakers/osim.wav data/out/a1_003.wav --json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.SpeakerSame import SpeakerSame


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Biometric Speaker Verification: Evaluates if a test audio matches a reference speaker voice (0-100 score)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "speaker_wav",
        type=str,
        help="Path to reference speaker audio file (.wav or .mp3)",
    )
    parser.add_argument(
        "test_wav",
        type=str,
        help="Path to test audio file (.wav or .mp3)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=75.0,
        help="Certainty threshold percentage (0-100) required to declare a match",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results in JSON format for automated pipelines",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run neural encoder on ('cuda', 'cpu', or auto-detect)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resemblyzer",
        choices=["resemblyzer", "wavlm"],
        help="Neural speaker verification encoder ('resemblyzer' or 'wavlm')",
    )
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    spk_path = Path(args.speaker_wav)
    test_path = Path(args.test_wav)

    if not spk_path.exists():
        for candidate_dir in [Path("data/out"), Path("speakers"), Path("data")]:
            cand = candidate_dir / spk_path
            if cand.exists():
                spk_path = cand
                break

    if not test_path.exists():
        for candidate_dir in [Path("data/out"), Path("speakers"), Path("data")]:
            cand = candidate_dir / test_path
            if cand.exists():
                test_path = cand
                break

    if not spk_path.exists():
        if args.json:
            print(json.dumps({"error": f"Speaker reference file not found: {spk_path}"}))
        else:
            print(f"Error: Speaker reference file not found: {spk_path}", file=sys.stderr)
        return 2

    if not test_path.exists():
        if args.json:
            print(json.dumps({"error": f"Test audio file not found: {test_path}"}))
        else:
            print(f"Error: Test audio file not found: {test_path}", file=sys.stderr)
        return 2

    try:
        verifier = SpeakerSame(device=args.device, encoder=args.encoder)
        res = verifier.compare(spk_path, test_path, threshold=args.threshold)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error during speaker verification: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        score = res["certainty"]
        cosine = res["cosine_similarity"]
        is_same = res["is_same"]
        thresh = res["threshold"]
        encoder_used = res.get("encoder", args.encoder)

        verdict_str = "MATCH (Same Speaker)" if is_same else "MISMATCH (Different Speaker)"
        symbol = "✅" if is_same else "❌"

        print(f"\n{'='*64}")
        print(f"🎙️  Speaker Verification: speaker-same")
        print(f"{'='*64}")
        print(f"Speaker Ref:      {spk_path}")
        print(f"Test Audio:       {test_path}")
        print(f"Encoder:          {encoder_used}")
        print(f"{'-'*64}")
        print(f"Certainty Score:  {score:.1f}%  (Threshold: >={thresh:.1f}%)")
        print(f"Cosine Sim:       {cosine:.4f}")
        print(f"Verdict:          {symbol} {verdict_str}")
        print(f"{'='*64}\n")

    return 0 if res["is_same"] else 1


if __name__ == "__main__":
    sys.exit(main())

