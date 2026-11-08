import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import Range, RangeList


def test_range_and_range_list():
    # 1. Range parsing
    r1 = Range.parse("5")
    assert r1.start == 5 and r1.end == 5
    assert str(r1) == "5"
    r2 = Range.parse("2-6")
    assert r2.start == 2 and r2.end == 6
    assert str(r2) == "2-6"
    # Inverted bounds auto-swap
    r_inv = Range.parse("6-2")
    assert r_inv.start == 2 and r_inv.end == 6
    # Wildcards: @-Y, x-Y, X-Y
    r_at_start = Range.parse("@-5")
    assert r_at_start.start is None and r_at_start.end == 5
    assert str(r_at_start) == "@-5"
    # Wildcards: Y-@, Y-x, Y-X
    r_at_end = Range.parse("10-@")
    assert r_at_end.start == 10 and r_at_end.end is None
    assert str(r_at_end) == "10-@"
    # Invalid formats rejected
    for invalid in ["-5", "5-", "abc", "0", "-1"]:
        try:
            Range.parse(invalid)
            assert False, f"Expected ValueError for '{invalid}'"
        except ValueError:
            pass
    # 2. Resolution and containment
    assert r_at_start.resolve(10) == (1, 5)
    assert r_at_end.resolve(10) == (10, 10)
    assert r2.resolve(10) == (2, 6)
    assert Range(15, 20).resolve(10) == (10, 10)
    assert Range().resolve(10) == (1, 10)
    assert Range().resolve(0) == (1, 1)
    assert Range(0, 15).resolve(10) == (1, 10)
    assert Range(-5, 0).resolve(10) == (1, 1)
    assert r2.resolve(0) == (1, 1)
    assert r2.contains(2) is True
    assert r2.contains(4) is True
    assert r2.contains(6) is True
    assert r2.contains(1) is False
    assert r2.contains(7) is False
    assert 4 in r2
    assert 1 not in r2
    # 3. RangeList parsing and normalization
    rl = RangeList.parse("10-12, 1-3, 5, 2-4, @-2")
    norm = rl.normalize(last=15)
    # Merges @-2, 1-3, 2-4, 5 -> [1, 5] and 10-12 -> [10, 12]
    assert len(norm.ranges) == 2
    assert norm.ranges[0] == Range(1, 5)
    assert norm.ranges[1] == Range(10, 12)
    # Contiguous range merging: [1, 3] and [4, 6] -> [1, 6]
    rl_cont = RangeList.parse("1-3, 4-6").normalize(last=10)
    assert len(rl_cont.ranges) == 1
    assert rl_cont.ranges[0] == Range(1, 6)
    # Wildcard lists with resolution
    rl_wild = RangeList.parse("@-3, 8-@, 5").normalize(last=10)
    assert len(rl_wild.ranges) == 3
    assert rl_wild.ranges[0] == Range(1, 3)
    assert rl_wild.ranges[1] == Range(5, 5)
    assert rl_wild.ranges[2] == Range(8, 10)
    # indices()
    assert rl_wild.indices(10) == [1, 2, 3, 5, 8, 9, 10]
    # contains()
    assert rl_wild.contains(1) is True
    assert rl_wild.contains(4) is False
    assert rl_wild.contains(5) is True
    assert rl_wild.contains(8) is True
    assert 3 in rl_wild
    assert 4 not in rl_wild
    print("[PASS] Range and RangeList verified.")


if __name__ == "__main__":
    test_range_and_range_list()
    print("\n🎉 ALL TESTS IN tests/test_range.py PASSED SUCCESSFULLY!")
