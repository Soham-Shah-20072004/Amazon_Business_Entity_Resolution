"""Text representations used by blocking and pair features.

Every record keeps several parallel views of its name/address (never one
"cleaned" string):

  norm   NFKC, lowercase, '&'->'and', punctuation->space (unicode kept)
  ascii  unidecode(norm): strips accents (Société->societe) and gives a rough
         Latin transliteration of non-Latin scripts (मुंबई->mumbii). Used as the
         main matching view so cross-script / accented variants can meet.
  canon  ascii with abbreviations mapped to one canonical token
         (pvt->private, rd->road, bd->boulevard ...)
  core   canon with legal-suffix tokens removed (name only)

Plus numeric tokens extracted from the address (house numbers, PIN/ZIP).
Abbreviation lists are hand-written domain knowledge, not external data.
"""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

_NON_ALNUM = re.compile(r"[^0-9a-z\s]+")
_WS = re.compile(r"\s+")
# split digit/letter boundaries so "12a" / "a12" / "no.12" compare token-wise
_DIGIT_ALPHA = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")

# canonical-token map; both sides of a pair are mapped, so direction is irrelevant
ABBREV = {
    # legal forms
    "pvt": "private", "pte": "private", "prv": "private",
    "ltd": "limited", "ltda": "limited", "lmt": "limited",
    "corp": "corporation", "co": "company", "cos": "company", "coy": "company",
    "inc": "incorporated", "incorp": "incorporated",
    "intl": "international", "int'l": "international",
    "mfg": "manufacturing", "mfrs": "manufacturers",
    "bros": "brothers", "assoc": "associates", "ent": "enterprises",
    "svcs": "services", "svc": "services", "sys": "systems", "tech": "technologies",
    "natl": "national", "dept": "department", "univ": "university",
    "hosp": "hospital", "med": "medical", "pharma": "pharmaceuticals",
    # address words (US / India / France)
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place", "sq": "square",
    "pkwy": "parkway", "hwy": "highway", "fwy": "freeway", "expy": "expressway",
    "cir": "circle", "trl": "trail", "ter": "terrace", "aly": "alley",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "rm": "room", "pob": "po",
    "nr": "near", "opp": "opposite", "bhd": "behind", "nagr": "nagar", "ngr": "nagar",
    "mkt": "market", "stn": "station", "sec": "sector", "ph": "phase",
    "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk",
    "no": "number", "num": "number", "nos": "number",
    "soc": "society", "apts": "apartment", "cplx": "complex",
    "r": "rue", "imp": "impasse", "che": "chemin", "rte": "route", "fbg": "faubourg",
}

LEGAL_TOKENS = frozenset({
    "private", "limited", "corporation", "company", "incorporated", "llc", "llp",
    "pllc", "plc", "lp", "gmbh", "ag", "sa", "sas", "sasu", "sarl", "eurl", "sci",
    "snc", "scop", "the", "opc", "pc", "pa", "dba", "group",
})

# US state / Indian state names are left as-is; common words that should not
# drive retrieval are handled by IDF weighting instead of a stop list.


def norm(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("&", " and ").replace("@", " at ")
    text = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return _WS.sub(" ", text).strip()


def to_ascii(normed: str) -> str:
    if not normed:
        return ""
    text = unidecode(normed).lower()
    text = _NON_ALNUM.sub(" ", text)
    text = _DIGIT_ALPHA.sub(" ", text)
    return _WS.sub(" ", text).strip()


def canon(ascii_text: str) -> str:
    if not ascii_text:
        return ""
    return " ".join(ABBREV.get(t, t) for t in ascii_text.split())


def core_name(canon_text: str) -> str:
    toks = [t for t in canon_text.split() if t not in LEGAL_TOKENS]
    return " ".join(toks) if toks else canon_text


def acronym(canon_text: str) -> str:
    toks = [t for t in canon_text.split() if t not in LEGAL_TOKENS and not t.isdigit()]
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


_NUM_TOKEN = re.compile(r"\d+")


def numbers(ascii_addr: str) -> list[str]:
    """Digit runs in order of appearance, leading zeros stripped ("007"->"7")."""
    return [n.lstrip("0") or "0" for n in _NUM_TOKEN.findall(ascii_addr)]


def postcodes(nums: list[str], raw_ascii: str) -> list[str]:
    """Postcode-like digit runs: 5 digits (US ZIP / FR code postal) or 6 (IN PIN).

    Also joins Indian "560 001" style splits. Country-agnostic on purpose.
    """
    out = [n for n in _NUM_TOKEN.findall(raw_ascii) if len(n) in (5, 6)]
    joined = re.findall(r"\b(\d{3})\s(\d{3})\b", raw_ascii)
    out += [a + b for a, b in joined]
    return list(dict.fromkeys(out))
