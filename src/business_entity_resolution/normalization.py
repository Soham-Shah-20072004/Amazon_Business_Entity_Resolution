"""Conservative baseline normalization for Stage 1.

Design rules (from the EDA plan):
  * Raw values are NEVER overwritten — normalization creates NEW columns.
  * The conservative form is the primary lexical representation.
  * The aggressive form exists ONLY for collision/recall analysis, never as a
    direct match rule.
  * We do NOT blindly remove house numbers, postal codes, business-type words
    or legal suffixes. Their behaviour must remain analyzable.

Unicode handling is deliberately conservative: we keep unicode letters
(e.g. French accents) instead of stripping everything to ASCII, because
France is unseen in training and must still be matchable.

Multilingual rules (§28):
  * Never translate to English; never ASCII-fold as "cleaning".
  * Transliteration is an ADDITIONAL representation only — never canonical,
    never a replacement for the raw/normalized Unicode text.
  * Country never implies language; no English-only stopword removal.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Analysis-only reference lists.
# These are used to *measure* suffix behaviour in EDA, NOT to strip fields in
# the conservative pipeline. The aggressive form uses them only so we can
# quantify over-normalization danger.
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_TOKENS = frozenset(
    {
        "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation",
        "co", "company", "corp.", "inc.", "ltd.", "llp", "pllc",
        "pvt", "private", "pte",
        "gmbh", "sarl", "sas", "sa", "sci", "eurl",  # seen in FR-style names
    }
)

BUSINESS_TYPE_TOKENS = frozenset(
    {
        "enterprises", "enterprise", "traders", "trading", "services",
        "solutions", "industries", "industry", "associates", "agency",
        "hotel", "hotels", "restaurant", "restaurants", "cafe", "bakery",
        "pharmacy", "hospital", "clinic", "store", "stores", "mart",
        "motors", "auto", "textiles", "jewellers", "jewelers",
    }
)

_WS_RE = re.compile(r"\s+")
# Numeric / alphanumeric tokens such as 12, 12A, B-17, 2/3, 4th, 560001.
_NUMERIC_TOKEN_RE = re.compile(r"[\w]*\d[\w]*(?:[/\-][\w]+)*", re.UNICODE)


def _collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_basic(value: object) -> str:
    """Conservative normalization.

    NFKC -> lowercase -> '&' to 'and' -> non-alphanumeric (unicode-aware) to
    space -> collapse whitespace. Digits, unicode letters, suffixes and word
    order are all preserved.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("&", " and ")
    # Keep unicode alphanumerics + whitespace; everything else becomes a space.
    text = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return _collapse_ws(text)


def normalize_aggressive(value: object) -> str:
    """Aggressive normalization for COLLISION ANALYSIS ONLY.

    conservative form -> drop legal-suffix + business-type tokens -> sort the
    remaining tokens -> dedupe. This intentionally destroys order and suffix
    evidence so EDA can measure how dangerous it would be as a match rule.
    """
    base = normalize_basic(value)
    if not base:
        return ""
    kept = [
        tok
        for tok in base.split(" ")
        if tok not in LEGAL_SUFFIX_TOKENS and tok not in BUSINESS_TYPE_TOKENS
    ]
    if not kept:
        return ""
    return " ".join(sorted(set(kept)))


def tokenize(text: str) -> List[str]:
    """Whitespace tokenize an (already normalized) string."""
    if not text:
        return []
    return [t for t in text.split(" ") if t]


def extract_numeric_tokens(address: object) -> List[str]:
    """Extract numeric/alphanumeric address tokens, preserving values.

    Keeps 12, 12A, 560001, 4th, 2/3, B-17, ... Comparison-friendly: lowercase.
    """
    if address is None:
        return []
    text = unicodedata.normalize("NFKC", str(address))
    toks = _NUMERIC_TOKEN_RE.findall(text)
    return [t.lower() for t in toks if t and t.strip("/-")]


def extract_postcode_like_tokens(address: object) -> List[str]:
    """Heuristic postcode/PIN-like tokens: all-digit tokens of length 5-6.

    This is a *country-agnostic heuristic* used for diagnostics, not a
    country-specific parser (France must work without special-casing).
    """
    return [t for t in extract_numeric_tokens(address) if t.isdigit() and len(t) in (5, 6)]


def string_stats(value: object) -> Dict[str, float]:
    """Character-level diagnostics for one raw string (country-shift EDA)."""
    text = "" if value is None else str(value)
    n = len(text)
    n_ascii = sum(1 for ch in text if ord(ch) < 128)
    digit_count = sum(1 for ch in text if ch.isdigit())
    punct_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    toks = text.split()
    return {
        "char_len": float(n),
        "digit_count": float(digit_count),
        "punct_count": float(punct_count),
        "non_ascii_ratio": float((n - n_ascii) / n) if n else 0.0,
        "token_count": float(len(toks)),
        "numeric_token_count": float(len(extract_numeric_tokens(text))),
    }


# ---------------------------------------------------------------------------
# Multilingual / multi-script support (§28).
#
# Script detection is APPROXIMATE: it recognizes the four scripts relevant to
# this pilot (Latin incl. Latin-1/Extended + Latin ligatures, Devanagari,
# Cyrillic, Arabic) plus scriptless-ASCII (digits/punct-only -> Latin,
# documented) and empty/missing. Anything else with a letter outside those
# ranges is "other Unicode". Use the SAME ranges for detection everywhere;
# they live here so the C-speed vectorized path (pandas regex) and the
# per-row path (ord lookups) cannot drift apart.
# ---------------------------------------------------------------------------

# Codepoint ranges per script (inclusive). Latin covers ASCII letters,
# Latin-1 Supplement letters (excludes × ÷ and control-adjacent symbols via
# the ranges below), Latin Extended A/B, Latin Extended Additional, and the
# Latin ligatures in Alphabetic Presentation Forms.
_LATIN_ORD_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x41, 0x5A), (0x61, 0x7A),
    (0xC0, 0xD6), (0xD8, 0xF6), (0xF8, 0xFF),
    (0x100, 0x24F),
    (0x1E00, 0x1EFF),
    (0xFB00, 0xFB06),
)
_DEVANAGARI_ORD_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0900, 0x097F), (0xA8E0, 0xA8FF),
)
_CYRILLIC_ORD_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0400, 0x04FF), (0x0500, 0x052F),
)
_ARABIC_ORD_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
)

_SCRIPT_ORD_RANGES = {
    "Latin": _LATIN_ORD_RANGES,
    "Devanagari": _DEVANAGARI_ORD_RANGES,
    "Cyrillic": _CYRILLIC_ORD_RANGES,
    "Arabic": _ARABIC_ORD_RANGES,
}


def _ranges_to_regex(ranges: Tuple[Tuple[int, int], ...]) -> str:
    return "[" + "".join(f"\\u{lo:04X}-\\u{hi:04X}" for lo, hi in ranges) + "]"


# Single source for the vectorized path: script "contains" patterns plus one
# negated-class pattern catching letters outside all recognized ranges.
SCRIPT_REGEX: Dict[str, str] = {
    key: _ranges_to_regex(ranges) for key, ranges in _SCRIPT_ORD_RANGES.items()
}
_RECOGNIZED_CLASSES = "".join(
    f"\\u{lo:04X}-\\u{hi:04X}"
    for ranges in _SCRIPT_ORD_RANGES.values()
    for lo, hi in ranges
)
# Word-char (letter/digit/underscore in Unicode terms) outside the recognized
# ranges, excluding decimal digits and the underscore itself. One
# str.contains pass, no large temporaries.
SCRIPT_REGEX["other"] = "[^\\W\\d_" + _RECOGNIZED_CLASSES + "]"
SCRIPT_CATEGORIES: Tuple[str, ...] = (
    "Latin", "Devanagari", "Cyrillic", "Arabic",
    "mixed", "other Unicode", "empty/missing",
)


def _in_ranges(ord_value: int, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    return any(lo <= ord_value <= hi for lo, hi in ranges)


def detect_script_category(value: object) -> str:
    """Classify one raw string into a coarse script bucket (§28.2).

    Single-script strings return that script; strings mixing >= 2 recognized
    scripts (or mixing a recognized script with other Unicode letters) are
    "mixed". Pure digits/punctuation/whitespace (no letters at all) default
    to "Latin" as documented ("scriptless ASCII"); symbol-only non-ASCII
    strings (emoji etc.) are "other Unicode". Mirrors the vectorized path.
    """
    text = "" if value is None else str(value)
    if not text.strip():
        return "empty/missing"
    has_latin = has_deva = has_cyrl = has_arab = has_other = False
    for ch in text:
        o = ord(ch)
        if _in_ranges(o, _LATIN_ORD_RANGES):
            has_latin = True
        elif _in_ranges(o, _DEVANAGARI_ORD_RANGES):
            has_deva = True
        elif _in_ranges(o, _CYRILLIC_ORD_RANGES):
            has_cyrl = True
        elif _in_ranges(o, _ARABIC_ORD_RANGES):
            has_arab = True
        elif ch.isalpha():
            has_other = True
        if has_latin + has_deva + has_cyrl + has_arab + has_other >= 2:
            return "mixed"
    if has_latin:
        return "Latin"
    if has_deva:
        return "Devanagari"
    if has_cyrl:
        return "Cyrillic"
    if has_arab:
        return "Arabic"
    if has_other:
        return "other Unicode"
    # No letters at all: scriptless ASCII -> Latin, else other Unicode.
    return "Latin" if all(ord(c) < 128 for c in text) else "other Unicode"


def pair_script_bucket(script_a: str, script_b: str) -> str:
    """Coarse script-pair bucket for matched pairs (§28.5)."""
    if script_a == "Latin" and script_b == "Latin":
        return "latin_latin"
    deva = {"Devanagari"}
    if script_a in deva and script_b in deva:
        return "deva_deva"
    if (script_a in deva) != (script_b in deva) and (
        script_a == "Latin" or script_b == "Latin"
    ):
        return "cross_latin_deva"
    if script_a == script_b and script_a not in ("mixed", "empty/missing"):
        return "same_script_other"
    if script_a != script_b and "mixed" not in (script_a, script_b):
        return "cross_other"
    return "mixed_or_uninformative"


# ---------------------------------------------------------------------------
# Measurement-only token lists (§28.6). NEVER used to strip text; only to
# *audit* what token-based features would do to non-Latin strings.
# ---------------------------------------------------------------------------

# Candidate Indic legal-suffix tokens OBSERVED in the pilot data. Listing them
# here asserts NO semantic equivalence with (or among) each other or with
# English suffixes — EDA only measures their overlap on positive pairs.
INDIC_SUFFIX_TOKENS = frozenset({"प्राइवेट", "लिमिटेड", "एलएलपी", "प्रा", "लि"})

# Candidate English stopwords + conjunction symbols, used ONLY to measure
# overlap quality (stopword-only overlap) on positive pairs. Never removed.
EN_STOPWORDS_MEASURE = frozenset(
    {"and", "the", "of", "for", "a", "an", "at", "in", "on", "&"}
)


def normalize_unicode(value: object) -> str:
    """NFKC + whitespace-collapse only. Preserves case, punctuation, scripts.

    This is the "unicode-norm" representation: compatibility-equivalent forms
    (half/full-width, ligatures, composed/decomposed accents) unify, while
    everything else — including non-Latin scripts — is untouched.
    """
    if value is None:
        return ""
    return _collapse_ws(unicodedata.normalize("NFKC", str(value)))


# ---------------------------------------------------------------------------
# Transliteration backends (§28.1/§28.4).
#
# Transliteration is an ADDITIONAL FEATURE, never canonical: outputs feed EDA
# comparison columns only. Backend preference: "auto" (use Unidecode when
# installed, else builtin) | "unidecode" | "builtin" | "none".
#
# The builtin backend is deterministic and dependency-free:
#   * Devanagari runs -> ITRANS-ish ASCII via a data-driven map built from
#     unicodedata names (independent vowels, consonants with inherent "a",
#     matras, virama/conjuncts, anusvara/chandrabindu -> n, visarga -> h,
#     nukta variants incl. decomposed consonant+nukta, digits, danda->space).
#     Approximations: schwa is ALWAYS emitted (no Hindi schwa deletion),
#     anusvara is always "n" (no homorganic nasal), no dictionary lookup.
#   * Other scripts: NFKD accent folding only (é->e); characters with no
#     decomposition (Cyrillic, Arabic, CJK, ...) pass through unchanged.
# ASCII-only input is returned unchanged by every backend.
# ---------------------------------------------------------------------------

_DEVA_BASE_TRANLIT = {
    # Independent vowels / vowel-sign names share transliterations.
    "SHORT A": "a", "A": "a", "AA": "aa", "I": "i", "II": "ii",
    "U": "u", "UU": "uu", "VOCALIC R": "ri", "VOCALIC RR": "rii",
    "VOCALIC L": "li", "VOCALIC LL": "lii", "CANDRA E": "e",
    "SHORT E": "e", "E": "e", "AI": "ai", "CANDRA O": "o",
    "SHORT O": "o", "O": "o", "AU": "au", "CANDRA A": "a",
    # Consonants.
    "KA": "k", "KHA": "kh", "GA": "g", "GHA": "gh", "NGA": "ng",
    "CA": "ch", "CHA": "chh", "JA": "j", "JHA": "jh", "NYA": "ny",
    "TTA": "T", "TTHA": "Th", "DDA": "D", "DDHA": "Dh", "NNA": "N",
    "TA": "t", "THA": "th", "DA": "d", "DHA": "dh", "NA": "n",
    "NNNA": "n", "PA": "p", "PHA": "ph", "BA": "b", "BHA": "bh",
    "MA": "m", "YA": "y", "RA": "r", "RRRA": "r", "LA": "l",
    "LLA": "L", "LLLA": "L", "VA": "v", "SHA": "sh", "SSHA": "shh",
    "SA": "s", "HA": "h", "QA": "q", "KHHA": "khh", "GHHA": "gh",
    "ZA": "z", "DDDHA": "Rh", "RHA": "Rh", "FA": "f", "YYA": "y",
    "GLOTTAL STOP": "q",
}
_DEVA_CONSONANT_NAMES = frozenset({
    "KA", "KHA", "GA", "GHA", "NGA", "CA", "CHA", "JA", "JHA", "NYA",
    "TTA", "TTHA", "DDA", "DDHA", "NNA", "TA", "THA", "DA", "DHA",
    "NA", "NNNA", "PA", "PHA", "BA", "BHA", "MA", "YA", "RA", "RRRA",
    "LA", "LLA", "LLLA", "VA", "SHA", "SSHA", "SA", "HA", "QA",
    "KHHA", "GHHA", "ZA", "DDDHA", "RHA", "FA", "YYA", "GLOTTAL STOP",
})
_DEVA_SIGN_TRANLIT = {
    "CANDRABINDU": "n", "INVERTED CANDRABINDU": "n", "ANUSVARA": "n",
    "VISARGA": "h", "AVAGRAHA": "'", "UDATTA": "", "ANUDATTA": "",
    "GRAVE": "", "ACUTE": "", "HIGH SPACING DOT": ".",
    "ABBREVIATION SIGN": "",
}
_DEVA_DIGITS = {
    "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
    "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9",
}
_DEVA_VIRAMA = "\u094d"
_DEVA_NUKTA = "\u093c"


def _build_devanagari_tables():
    """Build transliteration tables from unicodedata names (import time)."""
    consonants: Dict[str, str] = {}
    standalone: Dict[str, str] = {}
    vowel_signs: Dict[str, str] = {}
    for codepoint in range(0x0900, 0x0980):
        ch = chr(codepoint)
        name = unicodedata.name(ch, "")
        if name.startswith("DEVANAGARI LETTER "):
            key = name[len("DEVANAGARI LETTER "):]
            translit = _DEVA_BASE_TRANLIT.get(key)
            if translit is None:
                continue
            if key in _DEVA_CONSONANT_NAMES:
                consonants[ch] = translit
            else:
                standalone[ch] = translit
        elif name.startswith("DEVANAGARI VOWEL SIGN "):
            key = name[len("DEVANAGARI VOWEL SIGN "):]
            translit = _DEVA_BASE_TRANLIT.get(key)
            if translit is not None:
                vowel_signs[ch] = translit
        elif name.startswith("DEVANAGARI SIGN "):
            key = name[len("DEVANAGARI SIGN "):]
            if key in ("VIRAMA", "NUKTA"):
                continue  # handled structurally in the loop
            if key in _DEVA_SIGN_TRANLIT:
                standalone[ch] = _DEVA_SIGN_TRANLIT[key]
        elif name.startswith("DEVANAGARI DIGIT "):
            key = name[len("DEVANAGARI DIGIT "):]
            if key in _DEVA_DIGITS:
                standalone[ch] = _DEVA_DIGITS[key]
        elif name in ("DEVANAGARI DANDA", "DEVANAGARI DOUBLE DANDA"):
            standalone[ch] = " "
        elif name == "DEVANAGARI OM":
            standalone[ch] = "om"
    # Decomposed consonant + nukta sequences (e.g. क + nukta -> q).
    nukta_fix: Dict[str, str] = {}
    for letter_name, translit in (
        ("KA", "q"), ("KHA", "khh"), ("GA", "gh"), ("JA", "z"),
        ("DDA", "R"), ("DDHA", "Rh"), ("PHA", "f"), ("YA", "y"),
    ):
        try:
            nukta_fix[unicodedata.lookup(f"DEVANAGARI LETTER {letter_name}")] = translit
        except KeyError:
            continue
    return consonants, standalone, vowel_signs, nukta_fix


_DEVA_CONSONANTS, _DEVA_STANDALONE, _DEVA_VOWEL_SIGNS, _DEVA_NUKTA_FIX = (
    _build_devanagari_tables()
)


def _fold_char(ch: str) -> str:
    """NFKD accent fold for one char; unknown scripts pass through."""
    if ord(ch) < 128:
        return ch
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", ch)
        if unicodedata.category(c) != "Mn"
    )
    return folded if folded else ch


def _transliterate_builtin(text: str) -> str:
    if not text or text.isascii():
        return text
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _DEVA_CONSONANTS:
            base = _DEVA_CONSONANTS[ch]
            j = i + 1
            if j < n and text[j] == _DEVA_NUKTA:
                base = _DEVA_NUKTA_FIX.get(ch, base)
                j += 1
            if j < n and text[j] == _DEVA_VIRAMA:
                out.append(base)  # conjunct / vowel killed
                i = j + 1
                continue
            if j < n and text[j] in _DEVA_VOWEL_SIGNS:
                out.append(base + _DEVA_VOWEL_SIGNS[text[j]])
                i = j + 1
                continue
            out.append(base + "a")  # inherent schwa (never deleted)
            i = j
        elif ch in _DEVA_VOWEL_SIGNS:
            out.append(_DEVA_VOWEL_SIGNS[ch])  # stray matra
            i += 1
        elif ch in _DEVA_STANDALONE:
            out.append(_DEVA_STANDALONE[ch])  # vowels, signs, digits
            i += 1
        elif ch == _DEVA_VIRAMA or ch == _DEVA_NUKTA:
            i += 1  # stray control char
        else:
            out.append(_fold_char(ch))
            i += 1
    return "".join(out)


@functools.lru_cache(maxsize=8)
def resolve_transliteration_backend(preference: str = "auto") -> str:
    """Resolve a backend name to "unidecode" | "builtin" | "none".

    "auto" uses Unidecode when importable, else the builtin fallback.
    "unidecode" falls back to builtin with no exception when missing, so
    offline runs never crash. Results are cached (import attempts included).
    """
    pref = (preference or "auto").strip().lower()
    if pref in ("none", "off", "false", "no"):
        return "none"
    if pref == "builtin":
        return "builtin"
    try:
        import unidecode  # noqa: F401

        return "unidecode"
    except Exception:
        return "builtin" if pref in ("auto", "unidecode") else "builtin"


@functools.lru_cache(maxsize=262144)
def _transliterate_cached(text: str, backend: str) -> str:
    if backend == "unidecode":
        from unidecode import unidecode

        return str(unidecode(text))
    return _transliterate_builtin(text)


def transliterate_text(value: object, backend: str = "auto") -> str:
    """Transliterate one value (additional feature ONLY, never canonical).

    ASCII-only input is returned unchanged by every backend. Unknown scripts
    pass through the builtin backend unchanged (documented limitation).
    """
    if value is None:
        return ""
    text = str(value)
    if not text or text.isascii():
        return text
    resolved = resolve_transliteration_backend(backend)
    if resolved == "none":
        return text
    return _transliterate_cached(text, resolved)


# ---------------------------------------------------------------------------
# Character n-gram helpers (§28.7). Whitespace-insensitive by design (spaces
# are unreliable across scripts); short strings fall back to the whole
# string as one gram so nothing vanishes silently.
# ---------------------------------------------------------------------------

def char_ngram_set(text: str, sizes: Tuple[int, ...] = (3, 4, 5)) -> Set[str]:
    """Unique character n-grams over whitespace-stripped text."""
    compact = re.sub(r"\s+", "", text or "")
    grams: Set[str] = set()
    if not compact:
        return grams
    for size in sizes:
        if size <= 0:
            continue
        if len(compact) < size:
            grams.add(compact)
            continue
        for i in range(len(compact) - size + 1):
            grams.add(compact[i:i + size])
    return grams


def jaccard_similarity(first: Set[str], second: Set[str]) -> float:
    """Jaccard similarity; both-empty -> 1.0, one-empty -> 0.0."""
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def char_ngram_jaccard(
    first: str, second: str, sizes: Tuple[int, ...] = (3, 4, 5)
) -> float:
    """Character n-gram Jaccard between two strings."""
    return jaccard_similarity(
        char_ngram_set(first or "", sizes), char_ngram_set(second or "", sizes)
    )


def word_jaccard(first: str, second: str) -> float:
    """Whitespace-token Jaccard between two (already normalized) strings."""
    return jaccard_similarity(set(tokenize(first or "")), set(tokenize(second or "")))
