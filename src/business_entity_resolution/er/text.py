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

# canonical-token maps; both sides of a pair are mapped, so direction is irrelevant.
# Names and addresses get separate maps ("co" = company in a name, Colorado in an address).
NAME_ABBREV = {
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
}

ADDR_ABBREV = {
    # street words (US / India / France)
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
    # old / alternate city names seen in the data (e.g. Mumbai <-> Greater Bombay)
    "bombay": "mumbai", "bangalore": "bengaluru", "madras": "chennai", "calcutta": "kolkata",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysore": "mysuru", "benaras": "varanasi", "banaras": "varanasi",
    # unidecoded Devanagari/Gujarati spellings of states & cities -> canonical code/name
    "mhaaraassttr": "mh", "dillii": "dl", "hriyaannaa": "hr", "gujraat": "gj", "krnaattk": "ka",
    "tmilnaaddu": "tn", "raajsthaan": "rj", "pnjaab": "pb", "bihaar": "br", "kerl": "kl",
    "telngaanaa": "ts", "munbii": "mumbai",
}

# state names -> postal codes (codes themselves are left unchanged). Multi-word
# names are replaced as phrases before tokenisation. Matching is within-country,
# so the few codes shared by a US and an Indian state (e.g. GA) never collide.
STATE_CODES = {
    # India
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl nd",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "uttr prdesh": "up", "mdhy prdesh": "mp", "pshcim bngaal": "wb",
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
_STATE_PHRASES = sorted((k for k in STATE_CODES if " " in k), key=len, reverse=True)
_STATE_WORDS = {k: v for k, v in STATE_CODES.items() if " " not in k}

LEGAL_TOKENS = frozenset({
    "private", "limited", "corporation", "company", "incorporated", "llc", "llp",
    "pllc", "plc", "lp", "gmbh", "ag", "sa", "sas", "sasu", "sarl", "eurl", "sci",
    "snc", "scop", "the", "opc", "pc", "pa", "dba", "group",
})

def norm(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("&", " and ").replace("@", " at ")
    # keep combining marks (category M*): Devanagari/Gujarati vowel signs and
    # viramas are not "alnum", and dropping them splits every word into letters
    text = "".join(ch if (ch.isalnum() or ch.isspace() or unicodedata.category(ch)[0] == "M")
                   else " " for ch in text)
    return _WS.sub(" ", text).strip()


# candra-O (ॉ/ऑ, Gujarati ૉ/ઑ, used for English "o" as in डॉक्टर) makes unidecode
# emit "on"; map it to the plain o-sign first so "doctor" meets "डॉक्टर"
_CANDRA_O = str.maketrans({"\u0949": "\u094b", "\u0911": "\u0913",
                           "\u0ac9": "\u0acb", "\u0a91": "\u0a93"})


def to_ascii(normed: str) -> str:
    if not normed:
        return ""
    text = unidecode(normed.translate(_CANDRA_O)).lower()
    text = _NON_ALNUM.sub(" ", text)
    text = _DIGIT_ALPHA.sub(" ", text)
    return _WS.sub(" ", text).strip()


def canon_name(ascii_text: str) -> str:
    if not ascii_text:
        return ""
    return " ".join(NAME_ABBREV.get(t, t) for t in ascii_text.split())


def canon_addr(ascii_text: str) -> str:
    if not ascii_text:
        return ""
    text = f" {ascii_text} "
    for ph in _STATE_PHRASES:
        if f" {ph} " in text:
            text = text.replace(f" {ph} ", f" {STATE_CODES[ph]} ")
    out = []
    for t in text.split():
        t = ADDR_ABBREV.get(t, t)
        out.append(_STATE_WORDS.get(t, t))
    return " ".join(out)


_SKEL_SUBS = (("ph", "f"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "j"), ("w", "v"),
              ("nb", "mb"), ("np", "mp"), ("ee", "i"), ("oo", "u"))
_REPEAT = re.compile(r"(.)\1+")
_NON_INITIAL_H = re.compile(r"(?<=\w)h")
_NON_INITIAL_VOWEL = re.compile(r"(?<=\w)[aeiouy]")


def skeleton(ascii_text: str) -> str:
    """Rough sound skeleton so Latin and transliterated Indic spellings meet.

    unidecode gives 'dillii' for दिल्ली and 'gujraat' for गुजरात; both sides
    reduce to the same consonant frame: delhi/dillii -> 'dl', gujarat/gujraat
    -> 'gjrt', limited/limittedd -> 'lmtd'. Retrieval/feature view only.
    """
    out = []
    for tok in ascii_text.split():
        if tok.isdigit():
            out.append(tok)
            continue
        for a, b in _SKEL_SUBS:
            tok = tok.replace(a, b)
        tok = _NON_INITIAL_H.sub("", tok)
        tok = _REPEAT.sub(r"\1", tok)
        tok = _NON_INITIAL_VOWEL.sub("", tok)
        out.append(_REPEAT.sub(r"\1", tok))
    return " ".join(out)


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
