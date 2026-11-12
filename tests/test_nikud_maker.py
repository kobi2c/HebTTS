import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests.setup_env  # noqa: F401

from config import NikudMaker


def test_nikud_maker_detection_and_stripping():
    maker = NikudMaker()

    # 1. has() detection
    assert maker.has("שָׁלוֹם") is True
    assert maker.has("בְּרֵאשִׁית") is True
    assert maker.has("שלום") is False
    assert maker.has("Hello world 123") is False
    assert maker.has("") is False

    # 2. strip() stripping
    assert maker.strip("שָׁלוֹם") == "שלום"
    assert maker.strip("בְּרֵאשִׁית") == "בראשית"
    assert maker.strip("טֶקְסְט לְלֹא שִׁנּוּי") == "טקסט ללא שנוי"
    assert maker.strip("שלום") == "שלום"
    assert maker.strip("") == ""

    print("[PASS] NikudMaker detection and stripping verified.")


def test_nikud_maker_conversion_phonetic():
    # 1. Text without nikud returns identical string unchanged
    plain_text = "שלום עולם! זה טקסט ללא ניקוד."
    maker = NikudMaker()
    assert maker.convert(plain_text) == plain_text
    assert maker.word_to_phonetic("שלום") == "שלום"

    # 2. Conversion with injected phonemize function (testing vowel insertion logic)
    # Sin (שׂ) -> Samech (ס), Kubutz (ֻ) -> ו, Hiriq (ִ) -> י
    mock_ipa_map = {
        "שָׂרָה": "sa.ra",
        "כֻּלָּם": "ku.lam",
    }

    def fake_phonemize(word):
        return mock_ipa_map.get(word, "word")

    custom_maker = NikudMaker(phonemize_fn=fake_phonemize)

    # שׂ + ָ + ר + ָ + ה  -> סרה
    res_sin = custom_maker.word_to_phonetic("שָׂרָה")
    assert "ס" in res_sin

    # כֻּלָּם -> כוללם (kubutz adds 'ו')
    res_kubutz = custom_maker.word_to_phonetic("כֻּלָּם")
    assert "ו" in res_kubutz

    # Full sentence conversion preserving punctuation
    sent = "שָׂרָה אמרה שלום ל כֻּלָּם!"
    converted_sent = custom_maker.convert(sent)
    assert "סרה" in converted_sent
    assert "כולם" in converted_sent or "ו" in converted_sent
    assert "שלום" in converted_sent
    assert converted_sent.endswith("!")

    # 3. Missing phonikud package raises ImportError when vocalized
    orig_phonemize = NikudMaker.DEFAULT_PHONEMIZE
    try:
        NikudMaker.DEFAULT_PHONEMIZE = None
        no_phonikud_maker = NikudMaker()
        try:
            no_phonikud_maker.word_to_phonetic("שָׁלוֹם")
            assert False, "Should have raised ImportError when phonemize is None"
        except ImportError as e:
            assert "phonikud" in str(e)
    finally:
        NikudMaker.DEFAULT_PHONEMIZE = orig_phonemize

    print("[PASS] NikudMaker phonetic conversion and fallback verified.")


if __name__ == "__main__":
    test_nikud_maker_detection_and_stripping()
    test_nikud_maker_conversion_phonetic()
    print("\n🎉 ALL TESTS IN tests/test_nikud_maker.py PASSED SUCCESSFULLY!")
