#!/usr/bin/env python3
"""
speaker_verify.py - Alias for speaker_same.py
"""
import sys
from pathlib import Path

# Forward execution directly to speaker_same
from speaker_same import main

if __name__ == "__main__":
    sys.exit(main())
