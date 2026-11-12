import re
from typing import Callable, List, Tuple, Optional

try:
    from phonikud import phonemize as default_phonemize
except ImportError:
    default_phonemize = None


class NikudMaker:
    """
    Handles detection, stripping, and phonetic Ktiv-Male disambiguation
    of Hebrew vocalized text using phonetic phonemization rules.
    """

    NIKUD_RANGE = re.compile(r'[\u0591-\u05c7]')
    HEBREW_LETTER_PATTERN = re.compile(r'[\u05D0-\u05EA]')
    WORD_PATTERN = re.compile(r'[\u0590-\u05FF]+')
    DEFAULT_PHONEMIZE: Optional[Callable[[str], str]] = default_phonemize

    def __init__(self, phonemize_fn: Optional[Callable[[str], str]] = None) -> None:
        self._phonemize = phonemize_fn

    @property
    def phonemize_fn(self) -> Optional[Callable[[str], str]]:
        return self._phonemize if self._phonemize is not None else self.DEFAULT_PHONEMIZE

    def has(self, text: str) -> bool:
        """Returns True if the text contains any Hebrew cantillation or Nikud diacritics."""
        return bool(self.NIKUD_RANGE.search(text))

    def strip(self, text: str) -> str:
        """Strips all Hebrew cantillation and Nikud diacritics."""
        return self.NIKUD_RANGE.sub('', text)

    def word_to_phonetic(self, word: str) -> str:
        """
        Converts a single vocalized Hebrew word into disambiguated phonetic Ktiv-Male
        using phonikud IPA phonemes and diacritic patterns.
        """
        if not self.has(word):
            return word

        fn = self.phonemize_fn
        if fn is None:
            raise ImportError(
                "The 'phonikud' package is required for Nikud conversion. "
                "Please install it using: pip install phonikud"
            )

        # 1. Phonemize to get ground-truth modern IPA pronunciation
        ipa = fn(word)
        ipa_clean = ipa.replace("ˈ", "").replace("ˌ", "").replace(".", "")

        # Deconstruct letters and their associated diacritics
        letters_and_marks: List[Tuple[str, List[str]]] = []
        current_letter: Optional[str] = None
        current_marks: List[str] = []

        for ch in word:
            if '\u05D0' <= ch <= '\u05EA':
                if current_letter is not None:
                    letters_and_marks.append((current_letter, current_marks))
                current_letter = ch
                current_marks = []
            elif '\u0591' <= ch <= '\u05C7':
                current_marks.append(ch)

        if current_letter is not None:
            letters_and_marks.append((current_letter, current_marks))

        if not letters_and_marks:
            return self.strip(word)

        res: List[str] = []

        for idx, (let, marks) in enumerate(letters_and_marks):
            # 1. Sin (שׂ, U+05C2) pronounced as /s/ -> replace with Samech (ס)
            # In modern Hebrew, Sin is acoustically identical to Samech
            if let == 'ש' and '\u05C2' in marks:
                let = 'ס'
            res.append(let)

            next_let = letters_and_marks[idx + 1][0] if idx + 1 < len(letters_and_marks) else None
            is_word_end = (idx + 1 == len(letters_and_marks))

            # Diacritic flags
            has_kubutz = '\u05BB' in marks          # ֻ  (/u/)
            has_holam = '\u05B9' in marks           # ֹ  (/o/)
            has_kamatz_katan = '\u05C7' in marks    # ׇ  (/o/)
            has_hiriq = '\u05B4' in marks           # ִ  (/i/)

            # 2. Add 'ו' for /u/ or /o/ vowels if not already present
            if (has_kubutz or has_holam or has_kamatz_katan) and next_let != 'ו' and let != 'ו':
                res.append('ו')

            # 3. Add 'י' for /i/ vowels if not followed by 'י' and not at word end
            elif has_hiriq and next_let != 'י' and not is_word_end:
                res.append('י')

        return "".join(res)

    def convert(self, text: str) -> str:
        """
        Parses input text, finds words with Nikud, and converts them into
        disambiguated phonetic Ktiv-Male while preserving unvocalized text and punctuation.
        """
        if not self.has(text):
            return text

        # Tokenize by whitespace while preserving punctuation
        def replace_match(match: re.Match) -> str:
            token = match.group(0)
            if self.has(token):
                return self.word_to_phonetic(token)
            return token

        # Match tokens consisting of Hebrew letters and diacritics
        return self.WORD_PATTERN.sub(replace_match, text)
