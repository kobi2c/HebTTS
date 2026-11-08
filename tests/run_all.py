import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from tests.test_config import (
    test_enums,
    test_config_defaults,
    test_config_dicts,
    test_json_methods,
)
from tests.test_speaker import test_speaker_data
from tests.test_checkpoint import test_checkpoint_data
from tests.test_range import test_range_and_range_list
from tests.test_make_voice import (
    test_mki_parsing_with_config,
    test_voice_maker_args,
    test_voice_maker_range_only_run,
    test_voice_maker_format_options_run,
)
from tests.test_gradio_page import (
    test_generate_audio_strict_config,
    test_gradio_page_config_helpers,
    test_table_printing,
)

ALL_TESTS = [
    ("Enums", test_enums),
    ("Config Defaults", test_config_defaults),
    ("Config Dicts", test_config_dicts),
    ("JSON Serialization", test_json_methods),
    ("SpeakerStore / SpeakerData", test_speaker_data),
    ("CheckpointStore / CheckpointData", test_checkpoint_data),
    ("Range & RangeList", test_range_and_range_list),
    ("MKI Parsing & Text Chunking", test_mki_parsing_with_config),
    ("VoiceMaker CLI & Instance Fields", test_voice_maker_args),
    ("VoiceMaker --range-only Execution & Join", test_voice_maker_range_only_run),
    ("VoiceMaker MP3 / Both Output Formats", test_voice_maker_format_options_run),
    ("Strict Config Enforcement", test_generate_audio_strict_config),
    ("GradioPage Config Helpers", test_gradio_page_config_helpers),
    ("Table Printing", test_table_printing),
]


def run_all() -> int:
    print(f"\n{'='*70}")
    print(f"🧪  HebTTS Full Test Suite ({len(ALL_TESTS)} test suites)")
    print(f"{'='*70}\n")
    start = time.time()
    passed = 0
    failed = 0

    for name, test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {name} - {e}")
            failed += 1

    elapsed = time.time() - start
    print(f"\n{'-'*70}")
    print(f"Summary: {passed} passed, {failed} failed in {elapsed:.2f}s")
    print(f"{'-'*70}\n")

    if failed == 0:
        print("🎉 ALL TEST SUITES PASSED SUCCESSFULLY!\n")
        return 0
    else:
        print(f"❌ {failed} TEST SUITE(S) FAILED.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
