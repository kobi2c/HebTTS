import re
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union


@dataclass
class Range:
    """
    Represents an inclusive 1-based index interval [start, end].
    None indicates an open/unbounded boundary:
    - start=None: from the beginning (1)
    - end=None: through to the final element (last)
    """
    start: Optional[int] = None
    end: Optional[int] = None

    @classmethod
    def parse(cls, spec: str) -> "Range":
        """
        Parses a single range specification string.
        Supported formats:
        - 'Y'       -> single index (Range(Y, Y))
        - 'X-Y'     -> range from X to Y inclusive (auto-sorts if X > Y)
        - '@-Y'     -> from start to Y inclusive (Range(None, Y))
        - 'x-Y'     -> from start to Y inclusive (case-insensitive)
        - 'Y-@'     -> from Y to end inclusive (Range(Y, None))
        - 'Y-x'     -> from Y to end inclusive (case-insensitive)
        """
        s = spec.strip()
        if not s:
            raise ValueError("Empty range string cannot be parsed.")

        # 1. Single number: 'Y'
        if s.isdigit():
            val = int(s)
            if val < 1:
                raise ValueError(f"Range index must be positive (>= 1), got {val}")
            return cls(start=val, end=val)

        # 2. Open-ended start: '@-Y'
        m_start_wildcard = re.fullmatch(r"[@]-(\d+)", s)
        if m_start_wildcard:
            end_val = int(m_start_wildcard.group(1))
            if end_val < 1:
                raise ValueError(f"Range end must be positive (>= 1), got {end_val}")
            return cls(start=None, end=end_val)

        # 3. Open-ended end: 'Y-@'
        m_end_wildcard = re.fullmatch(r"(\d+)-[@]", s)
        if m_end_wildcard:
            start_val = int(m_end_wildcard.group(1))
            if start_val < 1:
                raise ValueError(f"Range start must be positive (>= 1), got {start_val}")
            return cls(start=start_val, end=None)

        # 4. Standard explicit range: 'X-Y'
        m_range = re.fullmatch(r"(\d+)-(\d+)", s)
        if m_range:
            v1 = int(m_range.group(1))
            v2 = int(m_range.group(2))
            if v1 < 1 or v2 < 1:
                raise ValueError(f"Range bounds must be positive (>= 1), got '{s}'")
            return cls(start=min(v1, v2), end=max(v1, v2))

        raise ValueError(
            f"Invalid range format: '{spec}'. Supported formats: 'X-Y', 'Y', '@-Y', 'x-Y', 'Y-@', 'Y-x'."
        )

    def resolve(self, last: int) -> Tuple[int, int]:
        """Resolves open bounds against a known last and clamps to [1, last]."""
        last = max(1, last)

        if self.start is None or self.start <= 0:
            s = 1
        elif self.start >= last:
            s = last
        else:
            s = self.start

        if self.end is None or self.end >= last:
            e = last
        elif self.end <= 0:
            e = 1
        else:
            e = self.end
        return (s, e) if s <= e else (e, s)

    def contains(self, index: int, last: Optional[int] = None) -> bool:
        """Checks if a 1-based index falls within this range."""
        if last is not None:
            s, e = self.resolve(last)
            return s <= index <= e

        if self.start is not None and index < self.start:
            return False
        if self.end is not None and index > self.end:
            return False
        return True

    def __contains__(self, index: int) -> bool:
        return self.contains(index)

    def __str__(self) -> str:
        if self.start is None and self.end is None:
            return "@-@"
        if self.start is None:
            return f"@-{self.end}"
        if self.end is None:
            return f"{self.start}-@"
        if self.start == self.end:
            return str(self.start)
        return f"{self.start}-{self.end}"


@dataclass
class RangeList:
    """Manages a collection of Range intervals with parsing, normalization, and lookup."""
    ranges: List[Range]

    def __init__(self, ranges: Optional[Sequence[Range]] = None) -> None:
        self.ranges = list(ranges) if ranges is not None else []

    @classmethod
    def parse(cls, spec: str) -> "RangeList":
        """
        Parses a comma-separated list of range specifications.
        Example: '1-3, 5, @-2, 10-@, 7-9'
        """
        if not spec or not spec.strip():
            return cls([])

        tokens = [t.strip() for t in spec.split(",") if t.strip()]
        parsed_ranges = [Range.parse(t) for t in tokens]
        return cls(parsed_ranges)

    def normalize(self, last: Optional[int] = None) -> "RangeList":
        """
        Normalizes ranges:
        - Resolves open bounds if last is given.
        - Sorts ranges by start.
        - Merges overlapping and contiguous intervals (e.g. [1, 3] + [4, 6] -> [1, 6]).
        """
        if not self.ranges:
            return RangeList([])

        if last is not None:
            # 1. Resolve all intervals against last
            resolved = [r.resolve(last) for r in self.ranges]
            resolved.sort(key=lambda item: (item[0], item[1]))

            merged: List[Tuple[int, int]] = []
            for s, e in resolved:
                if not merged:
                    merged.append((s, e))
                else:
                    prev_s, prev_e = merged[-1]
                    # Contiguous (prev_e + 1 >= s) or overlapping
                    if s <= prev_e + 1:
                        merged[-1] = (prev_s, max(prev_e, e))
                    else:
                        merged.append((s, e))

            return RangeList([Range(start=s, end=e) for s, e in merged])

        # last is None: normalize unbounded / bounded separately
        # Convert all to Range objects with start <= end
        clean_ranges: List[Range] = []
        has_wildcard_start = False
        max_wildcard_start_end = 0

        has_wildcard_end = False
        min_wildcard_end_start = float("inf")

        for r in self.ranges:
            if r.start is None and r.end is None:
                return RangeList([Range(None, None)])
            elif r.start is None:
                has_wildcard_start = True
                max_wildcard_start_end = max(max_wildcard_start_end, r.end or 1)
            elif r.end is None:
                has_wildcard_end = True
                min_wildcard_end_start = min(min_wildcard_end_start, r.start or 1)
            else:
                s, e = min(r.start, r.end), max(r.start, r.end)
                clean_ranges.append(Range(s, e))

        clean_ranges.sort(key=lambda r: (r.start or 0, r.end or 0))

        merged_bounded: List[Range] = []
        for r in clean_ranges:
            if not merged_bounded:
                merged_bounded.append(r)
            else:
                last = merged_bounded[-1]
                assert last.start is not None and last.end is not None
                assert r.start is not None and r.end is not None
                if r.start <= last.end + 1:
                    last.end = max(last.end, r.end)
                else:
                    merged_bounded.append(r)

        result: List[Range] = []
        if has_wildcard_start:
            result.append(Range(None, max_wildcard_start_end))

        result.extend(merged_bounded)

        if has_wildcard_end:
            result.append(Range(int(min_wildcard_end_start), None))

        return RangeList(result)

    def contains(self, index: int, last: Optional[int] = None) -> bool:
        """Returns True if the 1-based index is covered by any range in this list."""
        return any(r.contains(index, last) for r in self.ranges)

    def indices(self, last: int) -> List[int]:
        """Returns a sorted list of unique 1-based indices that match this RangeList."""
        normalized = self.normalize(last)
        matched: List[int] = []
        for r in normalized.ranges:
            s, e = r.resolve(last)
            matched.extend(range(s, e + 1))
        return sorted(set(matched))

    def __contains__(self, index: int) -> bool:
        return self.contains(index)

    def __iter__(self) -> Iterator[Range]:
        return iter(self.ranges)

    def __len__(self) -> int:
        return len(self.ranges)

    def __str__(self) -> str:
        return ", ".join(str(r) for r in self.ranges)
