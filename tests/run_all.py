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
from tests.test_chunk import test_chunk_and_chunk_list
from tests.test_make_voice import (
    test_mki_parsing_with_config,
    test_voice_maker_args,
    test_voice_maker_range_only_run,
    test_voice_maker_format_options_run,
    test_voice_maker_table_progress_output,
    test_voice_maker_chain_speaker_execution,
)
from tests.test_gradio_page import (
    test_generate_audio_strict_config,
    test_gradio_page_config_helpers,
    test_table_printing,
)
from tests.test_make_sample import (
    test_sampler_args,
    test_sampler_list_commands,
    test_sampler_run_execution,
)
from tests.test_engine import (
    test_voice_engine_abc,
    test_get_engine_factory,
    test_heb_tts_engine_create,
    test_voice_maker_with_custom_engine,
)

ALL_TESTS = [
    ("Enums", test_enums),
    ("Config Defaults", test_config_defaults),
    ("Config Dicts", test_config_dicts),
    ("JSON Serialization", test_json_methods),
    ("SpeakerStore", test_speaker_data),
    ("CheckpointStore", test_checkpoint_data),
    ("Range & RangeList", test_range_and_range_list),
    ("Chunk & ChunkList", test_chunk_and_chunk_list),
    ("MKI Parsing & Text Chunking", test_mki_parsing_with_config),
    ("VoiceMaker CLI & Instance Fields", test_voice_maker_args),
    ("VoiceMaker --range-only Execution & Join", test_voice_maker_range_only_run),
    ("VoiceMaker MP3 / Both Output Formats", test_voice_maker_format_options_run),
    ("VoiceMaker Structured Table Progress & Summary", test_voice_maker_table_progress_output),
    ("VoiceMaker Sequential Speaker Chaining", test_voice_maker_chain_speaker_execution),
    ("SampleMaker CLI & Instance Fields", test_sampler_args),
    ("SampleMaker List Commands", test_sampler_list_commands),
    ("SampleMaker Run Execution", test_sampler_run_execution),
    ("Strict Config Enforcement", test_generate_audio_strict_config),
    ("GradioPage Config Helpers", test_gradio_page_config_helpers),
    ("Table Printing", test_table_printing),
    ("VoiceEngine ABC Contract", test_voice_engine_abc),
    ("VoiceEngine Factory & Registration", test_get_engine_factory),
    ("HebTTSEngine Create & Token Stats", test_heb_tts_engine_create),
    ("VoiceMaker Pluggable Engine Synthesis", test_voice_maker_with_custom_engine),
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
            import traceback
            err_msg = str(e) if str(e) else type(e).__name__
            print(f"\n❌ FAIL: {name} ({err_msg})")
            traceback.print_exc()
            print()
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
