"""Name and address normalisation shared by blocking and feature engineering.

Everything here is deterministic string processing driven by small, hand-written
convention tables (street types, legal forms, state codes). No external reference
data, gazetteer or service is consulted.

Design notes (from EDA of the training data):
* Noise is injected per record: typos, leetspeak digits ("Wils0n"), random accents
  ("Córp"), junk prefixes ("***", "--", "<<", "##"), legal suffixes added / removed /
  moved to the front / bracketed, a spurious trailing word ("... Center"), duplicated
  words, word-order swaps, domain-style names ("schroederprinting.com"), and
  former-name aliases ("Vantagedova fka Pioneer Redwood L.L.C.").
* Addresses are reordered by comma component, abbreviated (St/Street, and even
  St -> "Saint"), state codes swap with full names or native-script names, house
  numbers get ranges/typos/leading zeros, and fillers like "null", "N/A" appear.
* Country is used only to pick convention tables; unknown countries fall back to the
  generic rules (the test set contains France, which training does not).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .translit import has_indic, transliterate

# --------------------------------------------------------------------------------------
# Convention tables
# --------------------------------------------------------------------------------------
LEGAL_FORMS: Dict[str, str] = {
    # English / US / India
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co", "cie": "co", "compania": "co",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lmited": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "pvtltd": "pvt",
    "public": "public", "plc": "plc",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "pa": "pa",
    "opc": "opc", "lllp": "lllp",
    # French
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "sca": "sca", "scs": "scs", "scop": "scop", "scp": "scp", "selarl": "selarl",
    "selas": "selas", "gie": "gie", "sem": "sem",
    # Other common international forms (open-set countries)
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa", "oy": "oy", "ab": "ab",
    # Legal words as they come out of `translit` for the nine Indic scripts
    # (प्राइवेट/പ്രൈവറ്റ്/பிரைவேட்/ପ୍ରାଇଭେଟ୍ -> praivet/praivatt/piraivet/praibhet, ...).
    "praivet": "pvt", "praivatt": "pvt", "piraivet": "pvt", "praibhet": "pvt", "praivat": "pvt",
    "limitet": "ltd", "limittad": "ltd", "limitad": "ltd", "limitted": "ltd",
    "kampani": "co", "kampni": "co", "karporeshan": "corp", "korporeshan": "corp",
}

# Honorific / filler tokens that carry no identity (removed from the name core only).
NAME_STOP = {"the", "and", "et", "of", "m/s", "ms", "messrs", "smt", "a", "an", "le", "la", "les",
             "de", "des", "du", "l", "d"}

# Street-type / address-word canonical forms. Keys are lower-case, dot-free tokens.
ADDR_CANON: Dict[str, str] = {
    # English street types
    "street": "st", "st": "st", "str": "st", "saint": "st", "stree": "st", "streeet": "st",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "avn": "ave", "avenu": "ave",
    "road": "rd", "rd": "rd", "raod": "rd",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd", "bld": "blvd",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct",
    "place": "pl", "pl": "pl", "plc": "pl",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy", "pkway": "pkwy",
    "highway": "hwy", "hwy": "hwy", "hiway": "hwy",
    "circle": "cir", "cir": "cir", "circ": "cir",
    "terrace": "ter", "ter": "ter", "terr": "ter",
    "trail": "trl", "trl": "trl",
    "square": "sq", "sq": "sq",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy",
    "turnpike": "tpke", "tpke": "tpke", "pike": "pike",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt", "pt": "pt",
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "apartments": "apt",
    "appartments": "apt", "apts": "apt", "unit": "unit", "floor": "fl", "fl": "fl", "flr": "fl",
    "building": "bldg", "bldg": "bldg", "room": "rm", "rm": "rm", "pobox": "pobox",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "ground": "g", "gf": "g",
    # India
    "sector": "sec", "sec": "sec", "sect": "sec", "nagar": "nagar", "ngr": "nagar",
    "marg": "marg", "extension": "extn", "extn": "extn", "ext": "extn", "colony": "colony",
    "col": "colony", "near": "near", "nr": "near", "opposite": "opp", "opp": "opp",
    "behind": "behind", "bh": "behind", "district": "dist", "dist": "dist", "distt": "dist",
    "taluk": "taluk", "taluka": "taluk", "tq": "taluk", "tal": "taluk", "post": "po",
    "cross": "cross", "main": "main", "layout": "layout", "phase": "phase", "ph": "phase",
    "block": "block", "blk": "block", "plot": "plot", "flat": "flat", "shop": "shop",
    "house": "house", "door": "door", "office": "office", "gali": "gali", "chowk": "chowk",
    "complex": "complex", "cmplx": "complex", "tower": "tower", "twr": "tower",
    # City aliases (old/new names used interchangeably in the data)
    "gurugram": "gurgaon", "bengaluru": "bangalore", "bombay": "mumbai", "calcutta": "kolkata",
    "madras": "chennai", "trivandrum": "thiruvananthapuram", "poona": "pune", "baroda": "vadodara",
    "mysuru": "mysore", "ahmadabad": "ahmedabad", "shadara": "shahdara", "shahdra": "shahdara",
}
# Country-specific abbreviations that would be wrong elsewhere (e.g. "R" = "Rue" only in
# France; in Indian addresses "R" is an initial). Keyed by lower-cased country label.
ADDR_CANON_BY_COUNTRY: Dict[str, Dict[str, str]] = {
    "france": {
        "rue": "rue", "r": "rue", "allee": "allee", "all": "allee", "chemin": "chemin",
        "ch": "chemin", "chem": "chemin", "impasse": "impasse", "imp": "impasse", "route": "rte",
        "rte": "rte", "faubourg": "fbg", "fbg": "fbg", "fg": "fbg", "quai": "quai",
        "cours": "cours", "crs": "cours", "residence": "res", "res": "res", "lieudit": "ld",
        "cedex": "", "bis": "", "ter": "", "quater": "",
    },
}

# Filler tokens that mean "no value" or are pure address boilerplate.
ADDR_STOP = {"null", "none", "nil", "na", "n/a", "no", "nos", "number", "the", "of", "and", "c/o",
             "co", "at", "de", "des", "du", "la", "le", "les", "d", "l", "h", "hn", "hno", "dno",
             "bis", "ter", "and"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr", "guam": "gu", "virgin islands": "vi",
}
INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "andaman and nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli and daman and diu": "dn", "daman and diu": "dd",
    "dadra and nagar haveli": "dn",
}
INDIA_CODE_ALIASES = {
    "ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "dh": "dn", "ua": "uk",
    # Native-script state names exactly as `translit` renders them. Only these 16 distinct
    # native-script address components occur in the training data (महाराष्ट्र, दिल्ली, ...).
    "maharashtr": "mh", "dilli": "dl", "karnatak": "ka", "tamizhnatu": "tn",
    "pashchimabang": "wb", "telangan": "tg", "hariyana": "hr", "keralan": "kl",
    "madhy pradesh": "mp", "andhrapradesh": "ap", "panjab": "pb", "orisha": "od",
}
FRANCE_REGIONS = {
    "auvergne rhone alpes": "ara", "bourgogne franche comte": "bfc", "bretagne": "bre",
    "centre val de loire": "cvl", "corse": "cor", "grand est": "ges", "hauts de france": "hdf",
    "ile de france": "idf", "normandie": "nor", "nouvelle aquitaine": "naq", "occitanie": "occ",
    "pays de la loire": "pdl", "provence alpes cote d azur": "pac",
}


def _state_table(names: Dict[str, str], extra_codes: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    table = dict(names)
    for code in set(names.values()):
        table[code] = code
    if extra_codes:
        table.update(extra_codes)
    return table


STATE_TABLES: Dict[str, Dict[str, str]] = {
    "us": _state_table(US_STATES),
    "india": _state_table(INDIA_STATES, INDIA_CODE_ALIASES),
    "france": {k: v for k, v in FRANCE_REGIONS.items()},
}

# --------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
_DOTTED_ABBR = re.compile(r"(?<![a-z0-9])((?:[a-z]\.){1,5}[a-z])\.?(?![a-z0-9])")
_ALIAS_SPLIT = re.compile(
    r"\s(?:f\s?/\s?k\s?/\s?a|fka|formerly(?:\s+known\s+as)?|d\s?/\s?b\s?/\s?a|dba|"
    r"doing\s+business\s+as|a\s?/\s?k\s?/\s?a|aka|also\s+known\s+as|t\s?/\s?a|trading\s+as)\b\s*:?\s*")
_DOMAIN = re.compile(r"^\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9.\-]*?)\s*\.\s*"
                     r"(?:com|net|org|in|co\.in|co|fr|biz|info|us|io)\s*$")
_BARE_COM = re.compile(r"^([a-z0-9]{6,})com$")
_NON_WORD = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")
_NAME_HASHNUM = re.compile(r"#\s*\d+")
_ADDR_TOKEN = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")
_DIGITS = re.compile(r"\d+")
_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_ADDR_NULLS = re.compile(r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b|\bnone\b")
_NO_PREFIX = re.compile(r"\b([a-z]{1,5})\s*\.\s*no\b\.?")      # "d.no." / "h.no" -> "d " / "h "
_POBOX = re.compile(r"\bp\s*\.?\s*o\s*\.?\s*box\b|\bpob\b")
# Soundex-style consonant classes (vowels, h, w, y are dropped). Coarse on purpose:
# Tamil script does not mark voicing (குளோபல் -> "kulopal" for "global") and
# Bengali/Odia write "v" with "bh", so b/p/f/v, c/g/j/k/s/z and d/t must collide.
_PHON_CLASS = str.maketrans({
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3", "l": "4", "m": "5", "n": "5", "r": "6",
    "a": None, "e": None, "i": None, "o": None, "u": None, "y": None, "h": None, "w": None,
})
_REPEAT = re.compile(r"(.)\1+")


def fold(text: str) -> Tuple[str, bool]:
    """NFKC -> Indic transliteration -> accent stripping -> lower case.

    Returns (folded_text, had_indic_script).
    """
    t = unicodedata.normalize("NFKC", text)
    indic = has_indic(t)
    if indic:
        t = transliterate(t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.lower(), indic


def phonetic_key(token: str) -> str:
    """Full-length Soundex-like code of a word: consonant classes, repeats collapsed,
    prefixed with '0' when the word starts with a vowel. Digits pass through, so numeric
    tokens keep their identity. "private"/"praivet"/"piraivet" -> "1613",
    "global"/"kulopal" -> "2414", "limited"/"limittad" -> "453"."""
    if not token:
        return token
    if token.isdigit():
        return token
    code = _REPEAT.sub(r"\1", token.translate(_PHON_CLASS))
    return ("0" + code) if token[0] in "aeiouy" else code


def _dedupe(tokens: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
@dataclass
class NameInfo:
    full: str            # all tokens, legal forms canonicalised
    core: str            # identity tokens only (legal forms + fillers removed)
    alias: str           # core of a former/trade name ("X fka Y" -> core(Y)); "" if none
    legal: str           # sorted canonical legal forms, space-joined
    is_domain: bool
    is_indic: bool


def _name_tokens(text: str) -> List[str]:
    t = text.replace("&", " and ").replace("+", " and ")
    t = _NAME_HASHNUM.sub(" ", t)
    t = _DOTTED_ABBR.sub(lambda m: m.group(1).replace(".", ""), t)
    t = t.replace("'", "").replace("’", "").replace("`", "")
    t = t.replace("m/s", " ")
    t = _NON_WORD.sub(" ", t)
    out = []
    for tok in t.split():
        if tok.isdigit():
            if len(tok) >= 4:          # "#31836"-style record numbers are noise
                continue
        elif not tok.isalpha():        # mixed letters+digits: undo leetspeak
            tok = tok.translate(_LEET)
        out.append(tok)
    return out


class Segmenter:
    """Splits concatenated names ("searsplatforms", "pediatricdentistryphysicians") into
    words with a unigram language model over a vocabulary built from the same split's
    Source-1 names (the provided reference records -- no external word list).

    Viterbi over all splits maximising the sum of word log-probabilities; a split is used
    only if the whole string is covered by >= 2 known words. Legal-form words are part of
    the vocabulary so a trailing suffix ("zfjclubsarl") separates too.
    """

    MAX_WORD = 24

    def __init__(self, counts: Dict[str, int], min_count: int = 2):
        total = float(sum(counts.values())) or 1.0
        self.logp = {w: math.log(c / total) for w, c in counts.items()
                     if c >= min_count and len(w) >= 2 and w.isalpha()}
        for lf in LEGAL_FORMS:
            self.logp.setdefault(lf, math.log(1.0 / total))

    def split(self, s: str) -> Optional[List[str]]:
        n = len(s)
        best = [-math.inf] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        logp = self.logp
        for i in range(1, n + 1):
            for j in range(max(0, i - self.MAX_WORD), i):
                if best[j] == -math.inf:
                    continue
                lp = logp.get(s[j:i])
                if lp is not None and best[j] + lp > best[i]:
                    best[i] = best[j] + lp
                    back[i] = j
        if best[n] == -math.inf:
            return None
        words, i = [], n
        while i > 0:
            words.append(s[back[i]:i])
            i = back[i]
        words.reverse()
        return words if len(words) >= 2 else None


def _split_legal(tokens: List[str]) -> Tuple[List[str], List[str]]:
    core, legal = [], []
    for tok in tokens:
        lf = LEGAL_FORMS.get(tok)
        if lf is not None:
            legal.append(lf)
        elif tok not in NAME_STOP:
            core.append(tok)
    return core, legal


def normalize_name(raw: str, segmenter: Optional["Segmenter"] = None) -> NameInfo:
    text, indic = fold(raw or "")
    text = _WS.sub(" ", text).strip()
    is_domain = False
    m = _DOMAIN.match(text)
    if m:
        text = m.group(1).replace(".", " ").replace("-", " ")
        is_domain = True
    else:
        m = _BARE_COM.match(text)
        if m and " " not in text:
            text = m.group(1)
            is_domain = True
    parts = _ALIAS_SPLIT.split(" " + text + " ", maxsplit=1)
    primary = parts[0]
    alias_raw = parts[1] if len(parts) > 1 else ""

    toks = _dedupe(_name_tokens(primary))
    if segmenter is not None and len(toks) == 1 and len(toks[0]) >= 8 and toks[0].isalpha() \
            and toks[0] not in segmenter.logp:
        pieces = segmenter.split(toks[0])     # concatenated / domain-style name
        if pieces:
            toks = _dedupe(pieces)
    core, legal = _split_legal(toks)
    if not core:                      # name made only of legal words: keep them as identity
        core = [t for t in toks if t not in NAME_STOP] or toks
    full = [LEGAL_FORMS.get(t, t) for t in toks]
    alias = ""
    if alias_raw:
        a_toks = _dedupe(_name_tokens(alias_raw))
        a_core, a_legal = _split_legal(a_toks)
        alias = " ".join(a_core)
        legal += a_legal
    return NameInfo(
        full=" ".join(full),
        core=" ".join(core),
        alias=alias,
        legal=" ".join(sorted(set(legal))),
        is_domain=is_domain,
        is_indic=indic,
    )


# --------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------
@dataclass
class AddrInfo:
    norm: str             # canonical tokens in original order ("" if empty)
    words: List[str]      # canonical non-numeric tokens (state included as its code)
    nums: List[str]       # digit runs, leading zeros stripped, in order of appearance
    codes: List[str]      # compact alnum form of every token that contains a digit
    state: str            # canonical state/region code, "" if not recognised
    empty: bool


def _country_key(country: str) -> str:
    return (country or "").strip().lower()


def _detect_state(component_tokens: List[str], ckey: str) -> str:
    """Return a state/region code if the whole comma-component names one, else ""."""
    table = STATE_TABLES.get(ckey)
    if not table or not component_tokens:
        return ""
    return table.get(" ".join(component_tokens), "")


_NO_WORD = re.compile(r"\bno\.|n°")
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_RUN_ZEROS = re.compile(r"(?<![0-9])0+(?=[0-9])")
_WORD_SPLIT = re.compile(r"[-/]")


def normalize_address(raw: str, country: str) -> AddrInfo:
    text, _ = fold(raw or "")
    text = _ADDR_NULLS.sub(" ", text)
    text = _POBOX.sub(" pobox ", text)
    text = _NO_PREFIX.sub(r"\1 ", text)
    text = _NO_WORD.sub(" ", text).replace("#", " ")
    ckey = _country_key(country)
    overlay = ADDR_CANON_BY_COUNTRY.get(ckey, {})
    words: List[str] = []
    nums: List[str] = []
    codes: List[str] = []
    norm: List[str] = []
    state = ""
    for comp in text.split(","):
        raw_words: List[str] = []     # pre-canonicalisation words, for state detection
        comp_words: List[str] = []
        comp_norm: List[str] = []
        has_number = False
        for tok in _ADDR_TOKEN.findall(comp):
            om = _ORDINAL.match(tok)
            if om:
                has_number = True
                n = om.group(1).lstrip("0") or "0"
                nums.append(n)
                codes.append(n)
                comp_norm.append(n)
                continue
            if any(ch.isdigit() for ch in tok):
                has_number = True
                # strip leading zeros of every digit run ("l053" == "l53", "xiii0447" == "xiii447")
                code = _RUN_ZEROS.sub("", _NON_ALNUM.sub("", tok)) or "0"
                codes.append(code)
                comp_norm.append(code)
                for d in _DIGITS.findall(tok):
                    nums.append(d.lstrip("0") or "0")
                continue
            for w in _WORD_SPLIT.split(tok.replace(".", "")):
                if not w:
                    continue
                raw_words.append(w)
                if w in ADDR_STOP:
                    continue
                w = overlay.get(w, ADDR_CANON.get(w, w))
                if w:
                    comp_words.append(w)
                    comp_norm.append(w)
        if not state and raw_words and not has_number:   # a state component has no numbers
            st = _detect_state(raw_words, ckey)
            if st:
                state = st
                words.append("state:" + st)
                norm.append(st)
                continue
        words.extend(comp_words)
        norm.extend(comp_norm)
    return AddrInfo(
        norm=" ".join(norm),
        words=words,
        nums=nums,
        codes=codes,
        state=state,
        empty=not norm,
    )
