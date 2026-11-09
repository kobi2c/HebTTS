import sys
import tempfile
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401
import soundfile as sf

from config import (
    Config,
    Format,
    RangeList,
    ChainMode,
    ChunkStatus,
    ChunkStats,
    Chunk,
    ChunkList,
    split_text,
    format_srt_time,
)


def test_chunk_status_int_enum():
    assert str(ChunkStatus.CREATED_S) == "CREATED_S"
    assert ChunkStatus.CREATED_S.name == "CREATED_S"
    assert str(ChunkStatus.CREATED_T) == "CREATED_T"
    assert ChunkStatus.CREATED_T.name == "CREATED_T"


    # String & integer lookup
    assert ChunkStatus("created_s") == ChunkStatus.CREATED_S
    assert ChunkStatus("created_t") == ChunkStatus.CREATED_T
    assert ChunkStatus("REUSED") == ChunkStatus.REUSED
    assert ChunkStatus(12) == ChunkStatus.CLEANED
    assert ChunkStatus(2 ) == ChunkStatus.CREATED_T
    print("[PASS] ChunkStatus IntEnum and string/int conversion verified.")


def test_chunk_stats():
    stats = ChunkStats(total=5)
    assert stats.total == 5
    assert stats.created == 0
    assert stats.created_s == 0
    assert stats.created_t == 0

    stats.record(ChunkStatus.CREATED_S)
    stats.record(ChunkStatus.CREATED_T)
    stats.record(ChunkStatus.REUSED)
    stats.record(ChunkStatus.SKIPPED)
    stats.record(ChunkStatus.CLEANED)
    stats.elapsed_sec = 1.23

    assert stats.created   == 2
    assert stats.created_s == 1
    assert stats.created_t == 1
    assert stats.reused    == 1
    assert stats.skipped   == 1
    assert stats.cleaned   == 1

    summary = stats.format_summary()
    assert summary == "Summary: 2 created (T: 1), 1 reused, 1 skipped (5 total chunks) in 1.2s"

    stats2 = ChunkStats(total=3, created_s=2, reused=1, elapsed_sec=0.5)
    assert stats2.format_summary() == "Summary: 2 created, 1 reused, 0 skipped (3 total chunks) in 0.5s"

    stats_clean = ChunkStats(total=3, cleaned=1, skipped=1, elapsed_sec=0.3)
    assert stats_clean.format_summary(clean_existing=True) == "Summary: 1 cleaned, 1 skipped (3 total chunks) in 0.3s"

    assert stats_clean.cleaned    == 1

    print("[PASS] ChunkStats recording, formatting, and dict access verified.")


def test_split_text_and_srt_formatting():
    # 1. format_srt_time
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(65.5) == "00:01:05,500"
    assert format_srt_time(3661.123) == "01:01:01,123"

    # 2. split_text
    long_text = (
        "היום אנחנו הולכים ללמוד על העולם הקסום של כללי ההתחלקות, "
        "אותם קיצורי דרך נחמדים שמצילים אותנו מהמטלה המייגעת לחלוטין של חילוק ארוך. "
        "האם יצא לכם להביט במספר ענק ולדעת מיד אם אפשר לפצל אותו בשלמות לקבוצות קטנות יותר?"
    )
    parts = split_text(long_text, max_len=100)
    assert len(parts) >= 2
    for p in parts:
        assert len(p) <= 100
    print("[PASS] split_text and format_srt_time verified.")


def test_chunk_paths_and_table_row():
    cfg = Config()
    c = Chunk(index=1, config=cfg, text="שלום עולם")
    assert c.status == ChunkStatus.PENDING
    assert c.duration == 0.0

    wav_p = c.wav_path("/tmp/out", "voice")
    txt_p = c.txt_path("/tmp/out", "voice")
    assert wav_p.name == "voice_001.wav"
    assert txt_p.name == "voice_001.txt"

    row = c.format_table_row(5)
    assert "[ 1/5 ]" in row
    assert "PENDING" in row
    assert "שלום עולם"[::-1] in row

    c.start_time = 0.0
    c.end_time = 2.5
    srt_block = c.to_srt_block(1)
    assert srt_block == "1\n00:00:00,000 --> 00:00:02,500\nשלום עולם\n\n"
    print("[PASS] Chunk paths, row formatting, and SRT block verified.")


def test_chunk_file_operations_and_clean():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        cfg = Config()
        chunk = Chunk(index=1, config=cfg, text="טקסט לבדיקה")

        assert not chunk.exists(out_dir, "test")

        # Clean non-existent file returns False and marks NOT_FOUND
        assert not chunk.clean(out_dir, "test")
        assert chunk.status == ChunkStatus.NOT_FOUND

        # Save audio and text
        sr = 24000
        test_audio = np.ones(sr, dtype=np.float32) * 0.5  # 1.0 second
        chunk.audio = test_audio
        chunk.sample_rate = sr
        chunk.save_audio(out_dir, "test")
        chunk.save_text(out_dir, "test")

        assert chunk.exists(out_dir, "test")
        assert (out_dir / "test_001.wav").exists()
        assert (out_dir / "test_001.txt").exists()
        assert (out_dir / "test_001.txt").read_text(encoding="utf-8") == "טקסט לבדיקה"

        # Load audio back
        chunk2 = Chunk(index=1, config=cfg, text="טקסט לבדיקה")
        loaded_audio = chunk2.load_audio(out_dir, "test")
        assert chunk2.status == ChunkStatus.REUSED
        assert np.isclose(chunk2.duration, 1.0)
        assert len(loaded_audio) == sr

        # Clean existing audio
        assert chunk2.clean(out_dir, "test", trim_noise=False, fade_ms=20.0, apply_fix=True)
        assert chunk2.status == ChunkStatus.CLEANED
    print("[PASS] Chunk file persistence, load, and clean verified.")


def test_chunk_list_collection_and_from_blocks():
    cfg1 = Config(predefined_speaker="osim")
    cfg2 = Config(predefined_speaker="shani")
    blocks = [
        (cfg1, "שורה ראשונה קצרה"),
        (cfg2, "שורה שנייה קצרה"),
    ]
    chunk_list = ChunkList.from_blocks(blocks, max_len=100)
    assert len(chunk_list) == 2
    assert chunk_list[0].index == 1
    assert chunk_list[0].config.predefined_speaker == "osim"
    assert chunk_list[0].text == "שורה ראשונה קצרה"
    assert chunk_list[1].index == 2
    assert chunk_list[1].config.predefined_speaker == "shani"

    # Slicing & iteration
    chunks_sub = chunk_list[0:1]
    assert len(chunks_sub) == 1
    assert [c.index for c in chunk_list] == [1, 2]

    # Target filtering
    active_range = RangeList.parse("2")
    targets, last_target = chunk_list.filter_targets(active_range, range_only=True)
    assert targets == {2}
    assert last_target == 2

    targets_all, last_all = chunk_list.filter_targets(active_range, range_only=False)
    assert targets_all == {1, 2}
    assert last_all == 2
    print("[PASS] ChunkList factory, slicing, and target filtering verified.")


def test_chunk_list_join_audio_and_save():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        sr = 24000
        c1 = Chunk(
            index=1,
            config=Config(pause_duration=0.5),
            text="שורה ראשונה",
            audio=np.ones(sr, dtype=np.float32) * 0.1,  # 1s
            status=ChunkStatus.CREATED_S,
            duration=1.0,
            start_time=0.0,
            end_time=1.0,
        )
        c2 = Chunk(
            index=2,
            config=Config(pause_duration=0.5),
            text="שורה שנייה",
            audio=np.ones(sr, dtype=np.float32) * 0.2,  # 1s
            status=ChunkStatus.CREATED_S,
            duration=1.0,
            start_time=1.5,
            end_time=2.5,
        )
        chunk_list = ChunkList([c1, c2], sample_rate=sr)

        # 1. Join audio
        merged_wav, total_sec = chunk_list.join_audio(target_indices={1, 2}, last_target_idx=2)
        assert merged_wav is not None
        # 1.0s + 0.5s pause + 1.0s = 2.5s = 60000 samples (no pause after chunk 2)
        expected_len = sr + int(round(sr * 0.5)) + sr
        assert len(merged_wav) == expected_len
        assert np.isclose(total_sec, 2.5)

        # 2. Save merged WAV & MP3
        saved = chunk_list.save_merged(out_dir, "merged", [Format.WAV, Format.MP3], merged_wav, sample_rate=sr)
        assert len(saved) == 2
        assert (out_dir / "merged.wav").exists()
        assert (out_dir / "merged.mp3").exists()

        # 3. Save SRT
        srt_path = chunk_list.save_srt(out_dir, "merged", target_indices={1, 2})
        assert srt_path.exists()
        srt_content = srt_path.read_text(encoding="utf-8")
        assert "1\n00:00:00,000 --> 00:00:01,000\nשורה ראשונה" in srt_content
        assert "2\n00:00:01,500 --> 00:00:02,500\nשורה שנייה" in srt_content
    print("[PASS] ChunkList join_audio, save_merged, and save_srt verified.")


def test_chunk_list_resolve_prompt_chaining():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        cfg1 = Config(predefined_speaker="osim")
        cfg2 = Config(predefined_speaker="osim")
        cfg3_diff = Config(predefined_speaker="shani")

        c1 = Chunk(index=1, config=cfg1, text="שורה ראשונה")
        c2 = Chunk(index=2, config=cfg2, text="שורה שנייה")
        c3 = Chunk(index=3, config=cfg3_diff, text="שורה שלישית")
        chunk_list = ChunkList([c1, c2, c3])

        # 1. Chunk 1 has no previous chunk -> (None, None)
        p_audio, p_txt = chunk_list.resolve_prompt(0, out_dir, "test", chain_speaker=ChainMode.Enabled)
        assert p_audio is None and p_txt is None

        # 2. Chunk 2 when chunk 1 files do not exist -> (None, None)
        p_audio, p_txt = chunk_list.resolve_prompt(1, out_dir, "test", chain_speaker=ChainMode.Enabled)
        assert p_audio is None and p_txt is None

        # Create chunk 1 files on disk
        c1_wav = out_dir / "test_001.wav"
        c1_txt = out_dir / "test_001.txt"
        sf.write(str(c1_wav), np.ones(24000, dtype=np.float32) * 0.1, 24000)
        c1_txt.write_text("שורה ראשונה", encoding="utf-8")

        # 3. Chunk 2 now resolves chunk 1's wav and text
        p_audio, p_txt = chunk_list.resolve_prompt(1, out_dir, "test", chain_speaker=ChainMode.Enabled)
        assert p_audio == str(c1_wav)
        assert p_txt == "שורה ראשונה"

        # 4. Chunk 2 when chain_speaker=ChainMode.Disabled (or default) -> (None, None)
        p_audio_no_chain, p_txt_no_chain = chunk_list.resolve_prompt(1, out_dir, "test", chain_speaker=ChainMode.Disabled)
        assert p_audio_no_chain is None and p_txt_no_chain is None
        p_audio_default, p_txt_default = chunk_list.resolve_prompt(1, out_dir, "test")
        assert p_audio_default is None and p_txt_default is None

        # 5. Chunk 3 has different speaker ("shani") -> (None, None)
        # Create chunk 2 files first
        c2_wav = out_dir / "test_002.wav"
        c2_txt = out_dir / "test_002.txt"
        sf.write(str(c2_wav), np.ones(24000, dtype=np.float32) * 0.1, 24000)
        c2_txt.write_text("שורה שנייה", encoding="utf-8")

        p_audio_diff, p_txt_diff = chunk_list.resolve_prompt(2, out_dir, "test", chain_speaker=True)
        assert p_audio_diff is None and p_txt_diff is None

    print("[PASS] ChunkList sequential prompt chaining resolution verified.")


def test_chunk_input_override():
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        cfg = Config()
        chunk = Chunk(index=1, config=cfg, text="טקסט מקורי")

        # 1. input_path verification
        inp_path = chunk.input_path(out_dir, "test")
        assert inp_path.name == "test_001.input.txt"
        assert not inp_path.exists()

        mock_vm = MagicMock()
        mock_model = MagicMock()
        mock_model.last_eos = (100, 200)
        mock_vm.page.load_model_if_needed.return_value = (mock_model, MagicMock(), MagicMock(), MagicMock())
        mock_vm.synthesize_chunk.return_value = np.ones(24000, dtype=np.float32) * 0.1

        # 2. Normal synthesize without .input.txt -> CREATED_S
        chunk.synthesize(mock_vm, out_dir, "test")
        assert chunk.status == ChunkStatus.CREATED_S
        assert chunk.text == "טקסט מקורי"
        assert (out_dir / "test_001.txt").read_text(encoding="utf-8") == "טקסט מקורי"

        # 3. Create .input.txt with IDENTICAL text -> still CREATED_S
        inp_path.write_text("טקסט מקורי", encoding="utf-8")
        chunk.synthesize(mock_vm, out_dir, "test")
        assert chunk.status == ChunkStatus.CREATED_S
        assert chunk.text == "טקסט מקורי"

        # 4. Create .input.txt with DIFFERENT text -> CREATED_T and chunk.text updated
        inp_path.write_text("טקסט שונה שנדרס", encoding="utf-8")
        chunk.synthesize(mock_vm, out_dir, "test")
        assert chunk.status == ChunkStatus.CREATED_T
        assert chunk.text == "טקסט שונה שנדרס"
        assert (out_dir / "test_001.txt").read_text(encoding="utf-8") == "טקסט שונה שנדרס"

        # Verify table row shows CREATED_T
        row = chunk.format_table_row(1)
        assert "CREATED_T" in row

    print("[PASS] Chunk text override from .input.txt verified.")


def test_chunk_and_chunk_list():
    test_chunk_status_int_enum()
    test_chunk_stats()
    test_split_text_and_srt_formatting()
    test_chunk_paths_and_table_row()
    test_chunk_file_operations_and_clean()
    test_chunk_list_collection_and_from_blocks()
    test_chunk_list_join_audio_and_save()
    test_chunk_list_resolve_prompt_chaining()
    test_chunk_input_override()


if __name__ == "__main__":
    test_chunk_and_chunk_list()
    print("\n🎉 ALL TESTS IN tests/test_chunk.py PASSED SUCCESSFULLY!")
