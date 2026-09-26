"""Rule-based transliteration of Indic (Brahmic) scripts to plain ASCII Latin.

About 23% of India Source-2 names and 12% of India Source-3 names are written in
Devanagari, Gujarati, Malayalam, Tamil, ... while Source 1 is always Latin, so the
matcher needs a common script. No external data or library is used: the nine
major Brahmic Unicode blocks (Devanagari U+0900 .. Malayalam U+0D00) share one
ISCII-derived layout -- the same offset inside each 128-codepoint block is the same
phonetic letter -- so a single offset table covers all of them.

The output is deliberately *matching-oriented*, not scholarly: long and short vowels
collapse (aa -> a, ii -> i), retroflex and dental stops collapse (ट/त -> t), and the
word-final inherent vowel is always dropped (लिमिटेड -> limited, बेस्ट -> best). Business
names are dominated by English loanwords, for which this is correct; for Sanskrit-origin
words (आदित्य -> adity) the dropped vowel is absorbed downstream by the phonetic key,
which ignores vowels anyway.
"""
from __future__ import annotations

import re

_BLOCK_STARTS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_INDIC_RE = re.compile("[ऀ-ൿ]")

# Consonants (offset -> Latin, inherent vowel added separately).
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # Devanagari/Bengali/Gurmukhi precomposed nukta forms (U+0958..U+095F etc.).
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
# Independent vowels.
_VOWELS = {
    0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
# Dependent vowel signs (matras): replace the inherent vowel.
_MATRAS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x4F: "au", 0x57: "au", 0x62: "li", 0x63: "li",
}
_VIRAMA = 0x4D
_NUKTA = 0x3C
_ANUSVARA = {0x01, 0x02}          # candrabindu, anusvara -> nasal
_VISARGA = 0x03
_NUKTA_MAP = {"k": "q", "kh": "kh", "g": "g", "j": "z", "ph": "f", "d": "r", "dh": "rh", "y": "y"}
_LABIALS = ("p", "b", "m")
# Characters outside the shared layout that still carry sound.
_SPECIAL = {
    0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k",  # Malayalam chillu
    0x0D4E: "r",                                                                    # Malayalam dot reph
    0x09CE: "t",                                                                    # Bengali khanda ta
    0x0A70: "n", 0x0A71: "",                                                        # Gurmukhi tippi, addak
    0x0B83: "",                                                                     # Tamil aytham
}


def has_indic(text: str) -> bool:
    return bool(_INDIC_RE.search(text))


def _offset(cp: int):
    for start in _BLOCK_STARTS:
        if start <= cp < start + 0x80:
            return cp - start
    return None


# Script-specific digraphs that the shared table cannot express.
_PRE_SUBS = (
    ("റ്റ", "ട്ട"),   # Malayalam റ്റ is "tt"
    ("ന്റ", "ന്ട"),   # Malayalam ന്റ is "nt"
    ("ற்ற", "ட்ர"),   # Tamil ற்ற is "tr"
)
_ANUSVARA_MARK = "\x00"


def transliterate(text: str) -> str:
    """Transliterate every Indic run in `text`; other characters pass through."""
    if not _INDIC_RE.search(text):
        return text
    for a, b in _PRE_SUBS:
        if a in text:
            text = text.replace(a, b)
    out: list[str] = []
    pending = False          # a consonant was emitted and its inherent 'a' is undecided
    prev_virama = False

    def flush_inherent(word_end: bool) -> None:
        nonlocal pending
        if pending:
            if not word_end:     # schwa deletion: never pronounce a word-final inherent 'a'
                out.append("a")
            pending = False

    for ch in text:
        cp = ord(ch)
        if cp in _SPECIAL:
            flush_inherent(False)
            out.append(_SPECIAL[cp])
            prev_virama = False
            continue
        off = _offset(cp)
        if off is None:
            if ch in "‌‍":          # ZWNJ / ZWJ: invisible joiners
                continue
            flush_inherent(True)
            out.append(ch)
            prev_virama = False
            continue
        if off in _CONS:
            flush_inherent(False)
            out.append(_CONS[off])
            pending = True
            prev_virama = False
        elif off == _NUKTA:
            # Modify the consonant just emitted (ज + ़ -> z, फ + ़ -> f, ...).
            if out and out[-1] in _NUKTA_MAP:
                out[-1] = _NUKTA_MAP[out[-1]]
        elif off in _MATRAS:
            pending = False
            out.append(_MATRAS[off])
            prev_virama = False
        elif off == _VIRAMA:
            pending = False          # explicit "no vowel"
            prev_virama = True
        elif off in _VOWELS:
            flush_inherent(False)
            out.append(_VOWELS[off])
            prev_virama = False
        elif off in _ANUSVARA:
            flush_inherent(False)
            out.append(_ANUSVARA_MARK)   # resolved to n/m below
            prev_virama = False
        elif off == _VISARGA:
            flush_inherent(False)
            out.append("h")
            prev_virama = False
        elif 0x66 <= off <= 0x6F:     # native digits
            flush_inherent(True)
            out.append(str(off - 0x66))
            prev_virama = False
        elif off in (0x64, 0x65):     # danda / double danda
            flush_inherent(True)
            out.append(" ")
            prev_virama = False
        else:                         # accents, avagraha, rare signs: silent
            continue
    flush_inherent(True)
    s = "".join(out)
    # Anusvara before a labial is pronounced 'm' (कंपनी -> kampani), otherwise 'n'.
    s = re.sub(_ANUSVARA_MARK + r"(?=[pbm])", "m", s)
    return s.replace(_ANUSVARA_MARK, "n")
