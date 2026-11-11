import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import (
    convert_nikud_text,
    has_nikud,
    strip_nikud,
    vocalized_word_to_phonetic,
    nikud_converter,
)


def test_nikud_detection_and_stripping():
    # 1. has_nikud detection
    assert has_nikud("שָׁלוֹם") is True
    assert has_nikud("בְּרֵאשִׁית") is True
    assert has_nikud("שלום") is False
    assert has_nikud("Hello world 123") is False
    assert has_nikud("") is False

    # 2. strip_nikud stripping
    assert strip_nikud("שָׁלוֹם") == "שלום"
    assert strip_nikud("בְּרֵאשִׁית") == "בראשית"
    assert strip_nikud("טֶקְסְט לְלֹא שִׁנּוּי") == "טקסט ללא שנוי"
    assert strip_nikud("שלום") == "שלום"
    assert strip_nikud("") == ""

    print("[PASS] Nikud detection and stripping verified.")


def test_nikud_conversion_phonetic():
    # 1. Text without nikud returns identical string unchanged
    plain_text = "שלום עולם! זה טקסט ללא ניקוד."
    assert convert_nikud_text(plain_text) == plain_text
    assert vocalized_word_to_phonetic("שלום") == "שלום"

    # 2. Conversion with mocked phonemize (testing vowel insertion logic)
    # Sin (שׂ) -> Samech (ס), Kubutz (ֻ) -> ו, Hiriq (ִ) -> י
    mock_ipa_map = {
        "שָׂרָה": "sa.ra",
        "כֻּלָּם": "ku.lam",
    }

    def fake_phonemize(word):
        return mock_ipa_map.get(word, "word")

    with patch.object(nikud_converter, "phonemize", side_effect=fake_phonemize):
        # שׂ + ָ + ר + ָ + ה  -> סרה
        res_sin = vocalized_word_to_phonetic("שָׂרָה")
        assert "ס" in res_sin

        # כֻּלָּם -> כוללם (kubutz adds 'ו')
        res_kubutz = vocalized_word_to_phonetic("כֻּלָּם")
        assert "ו" in res_kubutz

        # Full sentence conversion preserving punctuation
        sent = "שָׂרָה אמרה שלום ל כֻּלָּם!"
        converted_sent = convert_nikud_text(sent)
        assert "סרה" in converted_sent
        assert "כולם" in converted_sent or "ו" in converted_sent
        assert "שלום" in converted_sent
        assert converted_sent.endswith("!")

    # 3. Missing phonikud package raises ImportError when vocalized
    with patch.object(nikud_converter, "phonemize", None):
        try:
            vocalized_word_to_phonetic("שָׁלוֹם")
            assert False, "Should have raised ImportError when phonemize is None"
        except ImportError as e:
            assert "phonikud" in str(e)

    print("[PASS] Nikud phonetic conversion and fallback verified.")


if __name__ == "__main__":
    test_nikud_detection_and_stripping()
    test_nikud_conversion_phonetic()
    print("\n🎉 ALL TESTS IN tests/test_nikud_converter.py PASSED SUCCESSFULLY!")
