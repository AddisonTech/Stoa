"""Read each announcement page into one row per award, by fixed rules only.

A page is a run of paragraphs: a branch heading in capitals (ARMY, NAVY, AIR FORCE,
DEFENSE LOGISTICS AGENCY, ...), then one paragraph per award that opens with the awardee
("Shift5 Inc.,* Arlington, Virginia, was awarded a $99,000,000 ..."). A paragraph that
opens "The following companies" is followed by paragraphs listing the awardees.

Each award row has the announcement date, the branch, the company, its city and state,
the dollar value (the first dollar amount after the verb), the contract number (the first
contract number after the verb, a modification's base contract) and a type, by rules in
this order, the first that fits:

1. ``option``: an option is exercised ("exercising the second one-year option", or
   "option" in the award phrase, but not "with an option which if exercised").
2. ``modification``: the award phrase says "modification", "definitization" (not
   "undefinitized", which is a new letter contract) or "increases the ceiling"; or the text
   says "bringing the total cumulative face value", "this modification"; or it carries a
   modification number (P00018, A00002).
3. ``multiple_award``: more than one awardee, "multiple award", "each awarded", "will
   compete for".
4. ``idiq``: the award phrase names an indefinite-delivery/indefinite-quantity or
   requirements contract.
5. ``order``: the award phrase names a task, delivery, call or agreement order.
6. ``modification`` again: "previously awarded" with no order named (a task order "under
   a previously awarded" contract is new money).
7. ``new``: anything else that awards money (a definitive contract, a letter contract,
   an other transaction agreement).

``ceiling`` is True when the value is a ceiling or maximum ("ceiling", "maximum",
"not-to-exceed", "up to $"). Groups: option and modification are ``mod``,
multiple_award is ``multi``, the rest ``new``. No language model reads these pages.
"""

from __future__ import annotations

import html as htmllib
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from stoa.contracts.scrape import title_date

TYPES = ("new", "idiq", "order", "multiple_award", "modification", "option")
GROUP = {
    "new": "new",
    "idiq": "new",
    "order": "new",
    "multiple_award": "multi",
    "modification": "mod",
    "option": "mod",
}

STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming", "d.c.", "dc", "district of columbia", "puerto rico", "guam",
    "u.s. virgin islands", "virgin islands",
}  # fmt: skip

_DIV = re.compile(r"<(/?)div\b[^>]*>", re.I)
_SCRIPT = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)
_BLOCK_END = re.compile(
    r"</(?:p|div|li|h\d|td|tr|table|ul|ol)\s*>|<br\s*/?>|<p\b[^>]*>|<div\b[^>]*>", re.I
)
_TAG = re.compile(r"<[^>]+>")
_VERB = re.compile(
    r"\b(?:(?:has|have|had)\s+(?:also\s+)?been|was|were|is|are|will\s+be|has|have|had)\s+"
    r"(?:each\s+|jointly\s+|also\s+|collectively\s+)?(?:being\s+)?"
    r"(?:awarded|received|receiving|issued|sharing|been\s+awarded|award(?=\s+(?:a|an)\s))\b"
    r"|\b(?:received|receives|awarded)\s+(?:a|an)\s+(?=[^.]{0,80}\$)"
    r"|\bwill\s+(?:each\s+|all\s+)?(?:share|compete)\b"
    r"|\b(?:intends?|plans?)\s+to\s+award\b",
    re.I,
)
_PLURAL = re.compile(r"^\s*(?:were|are|have|will\s+be|will\s+(?:each\s+)?share)\b", re.I)
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s*(million|billion|thousand))?", re.I)
_PIID = re.compile(r"\b([A-Z][A-Z0-9]{5})-?(\d{2})-?([A-Z])-?([A-Z0-9]{4})\b")
_MOD_NO = re.compile(r"\b([PA]0{2,3}\d{2,3})\b")
_SOLICITATION = re.compile(r"solicitation(?:\s+number)?\s*(?:no\.\s*)?\(?\s*$", re.I)
_PAREN = re.compile(r"\(([^)]*)\)")
_FOLLOWING = re.compile(r"^\s*the following\b", re.I)

# An option exercised: "exercised an option", "is exercising options", "option exercise",
# or "option" in the award phrase itself. "an option which if exercised" on a new contract
# is not one.
_OPTION_TEXT = re.compile(
    r"\bexercis(?:e|es|ed|ing)\s+(?:of\s+)?(?:an?\s+|the\s+|its\s+|all\s+|[\w-]+\s+){0,3}"
    r"options?\b|\boptions?\s+(?:exercise|was exercised|were exercised|is being exercised|"
    r"has been exercised|have been exercised)\b",
    re.I,
)
_OPTION_PHRASE = re.compile(r"\boptions?\b", re.I)
# Options a new contract merely carries: "with an option", "options which if exercised".
_OPTION_MENTION = re.compile(
    r"\bwith (?:an? |\w+ |\w+-\w+ )?options?\b|\bif (?:all )?(?:the )?options?\b|"
    r"\bif exercised\b|\boptions? (?:which|that)\b|\bincluding (?:all )?options\b|"
    r"\boptions? periods?\b|\bbase\b[^.]{0,60}\boptions?\b",
    re.I,
)
# A modification: the award phrase says so, or the text does in words only a change to an
# existing contract uses. "aircraft modification services" deep in a new award does not count.
_MOD_PHRASE = re.compile(
    r"\bmodification\b|\bmodifies\b|\bmodify\b|(?<!un)definitiz|"
    r"increases? the (?:contract )?ceiling|ceiling increase|\bmod\b",
    re.I,
)
# "Bringing the total cumulative face value of the contract to $X" is a change to a contract;
# "This total cumulative face value is $200,000,000" on a new award is not.
_MOD_TEXT = re.compile(
    r"\bbringing the (?:total )?cumulative face value\b|"
    r"\bcumulative face value of the contract to\b|\bthis (?:contract )?modification\b|"
    r"\bthe modification\b|\bbringing the total\b|\bto (?:a |an )?existing contract\b",
    re.I,
)
# "Previously awarded" is weaker: a task order "under a previously awarded" contract is new
# money, so it counts as a modification only when the award phrase names no order.
_PREVIOUS = re.compile(r"previously[- ]awarded", re.I)
_PARENT = re.compile(r"\b(?:under|against|on)\s+(?:a|an|the|its|their|basic)\b", re.I)
_MULTI = re.compile(
    r"multiple[- ]award|\beach (?:been |be )?awarded\b|will compete for|\bcompete for each\b|"
    r"\bwere awarded\b|\bhave been awarded\b|\bare (?:being )?awarded\b",
    re.I,
)
_ORDER = re.compile(
    r"\b(?:task|delivery|call|agreement|purchase)[- ]orders?\b|\borders?\s*\([A-Z0-9-]{8,}\)|"
    r"\border\s+[A-Z][A-Z0-9]{5}-?\d{2}-?F|"
    r"\b(?:an?|the)\s+(?:[\w$,.-]+\s+){0,6}orders?\b(?!\s+to\b)",
    re.I,
)
_IDIQ = re.compile(
    r"indefinite[- ]delivery|indefinite[- ]quantity|\bIDIQ\b|requirements[- ]type|"
    r"requirements contract",
    re.I,
)
_CEILING = re.compile(r"\bceiling\b|\bmaximum\b|not[- ]to[- ]exceed|\bup to \$", re.I)


@dataclass
class Award:
    date: date
    article: int
    para: int
    branch: str
    company: str
    city: str
    state: str
    small_business: bool
    value: float | None
    contract: str
    mod_number: str
    type: str
    group: str
    ceiling: bool
    n_awardees: int
    text: str
    awardee_index: int = 0
    flags: list[str] = field(default_factory=list)


def clean(fragment: str) -> str:
    t = htmllib.unescape(_TAG.sub(" ", fragment))
    t = t.replace("\xa0", " ").replace(chr(0x2019), "'")
    t = t.replace(chr(0x201C), '"').replace(chr(0x201D), '"')
    return " ".join(t.split())


def body_html(page: str) -> str:
    """The article's body div, closed where its own nesting closes it."""
    i = page.find('<div class="body">')
    if i < 0:
        return ""
    depth = 0
    for m in _DIV.finditer(page, i):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return page[i : m.end()]
    return page[i:]


def paragraphs(page: str) -> list[str]:
    """The body's text blocks in order. Most pages put each award in a <p>; from late 2018 to
    mid 2019 they used <div>; so any block's end is a paragraph's end."""
    body = _SCRIPT.sub(" ", body_html(page))
    parts = _BLOCK_END.split(body)
    return [p for p in (clean(x) for x in parts) if p]


def is_heading(p: str) -> bool:
    letters = [c for c in p if c.isalpha()]
    if not letters or len(p) > 90 or _VERB.search(p):
        return False
    upper = sum(c.isupper() for c in letters) / len(letters)
    return upper > 0.85


def money(text: str) -> float | None:
    got = _MONEY.search(text)
    if not got:
        return None
    v = float(got.group(1).replace(",", "") or 0)
    mult = {"million": 1e6, "billion": 1e9, "thousand": 1e3}.get((got.group(2) or "").lower(), 1.0)
    return v * mult


def piid(text: str) -> str:
    """The first contract number in ``text`` that is not introduced as a solicitation; the
    first of any kind if all are."""
    found = list(_PIID.finditer(text))
    for got in found:
        if not _SOLICITATION.search(text[max(0, got.start() - 45) : got.start()]):
            return "".join(got.groups())
    return "".join(found[0].groups()) if found else ""


def award_phrase(rest: str) -> str:
    """From the verb to the first ' for ' or sentence end: what instrument was awarded.
    In "is awarded $23,873,454 for a firm-fixed-price contract" or "$9,975,624 for
    firm-fixed-price order N0001917F1568" the 'for' right after the amount is skipped."""
    start = 0
    m = _MONEY.search(rest)
    if m and re.match(r"\s+for\s", rest[m.end() :]):
        start = m.end() + 5
    cut = len(rest)
    for stop in (" for ", ". ", " to provide ", " to support "):
        k = rest.find(stop, start)
        if 0 <= k < cut:
            cut = k
    return rest[: min(cut, 400)]


def classify(text: str, n_awardees: int = 1, phrase: str | None = None) -> str:
    """The award type by the module's ordered rules. ``phrase`` is the award phrase
    (verb to the first ' for '); without one, the text's first 300 characters stand in."""
    phrase = phrase if phrase is not None else award_phrase(text[:300])
    if (
        _OPTION_PHRASE.search(phrase) and not _OPTION_MENTION.search(phrase)
    ) or _OPTION_TEXT.search(text):
        return "option"
    if _MOD_PHRASE.search(phrase) or _MOD_TEXT.search(text) or _MOD_NO.search(text):
        return "modification"
    # What was awarded is named before "under / against / on a ..." (the parent contract):
    # a task order "under a previously awarded multiple award IDIQ contract" is an order.
    core = _PARENT.split(phrase)[0]
    if n_awardees > 1 or _MULTI.search(core):
        return "multiple_award"
    if _IDIQ.search(core):
        return "idiq"
    if _ORDER.search(core):
        return "order"
    if _PREVIOUS.search(text):
        return "modification"
    return "new"


def split_place(lead: str) -> tuple[str, str, str, bool, list[str]]:
    """'Shift5 Inc.,* Arlington, Virginia' -> (name, city, state, small business, contracts)."""
    contracts = [piid(c) for c in _PAREN.findall(lead) if piid(c)]
    lead = _PAREN.sub(" ", lead)
    lead = _MONEY.sub(" ", lead)  # fuel lists give each awardee's amount: "$449,473,339"
    small = "*" in lead
    lead = lead.replace("*", "")
    lead = re.sub(r"\s+", " ", lead).strip(" ,;")
    parts = [x.strip() for x in lead.split(",") if x.strip()]
    if (
        len(parts) >= 3
        and parts[-1].lower() in ("d.c.", "dc")
        and parts[-2].lower() == "washington"
    ):
        return ", ".join(parts[:-2]), "Washington", "D.C.", small, contracts
    if len(parts) >= 3:
        return ", ".join(parts[:-2]), parts[-2], parts[-1], small, contracts
    if len(parts) == 2:
        if parts[1].lower() in STATES:
            return parts[0], "", parts[1], small, contracts
        return parts[0], parts[1], "", small, contracts
    return lead, "", "", small, contracts


def split_awardees(lead: str) -> list[str]:
    """Several awardees in one lead, by semicolons, else by ', and '."""
    lead = re.sub(r"^\s*(?:the following[^:]*:)", "", lead, flags=re.I)
    if ";" in lead:
        segs = re.split(r";\s*(?:and\s+)?", lead)
        # A stray semicolon ("Impact Components,*; San Diego, California (N68936-...)"):
        # when the others each carry a contract number, a piece without one joins the next.
        if sum("(" in s for s in segs) * 2 >= len(segs):
            merged: list[str] = []
            carry = ""
            for s in segs:
                if "(" not in s and s.strip(" ,;"):
                    carry += s.strip(" ;") + " "
                    continue
                merged.append(carry + s)
                carry = ""
            if carry:
                merged.append(carry)
            segs = merged
    else:
        segs = re.split(r",?\s+and\s+(?=[A-Z])", lead)
        # "Smith and Jones LLC, Austin, Texas" is one company: only split when each
        # piece carries its own place (at least two commas).
        if len(segs) > 1 and not all(s.count(",") >= 2 for s in segs):
            segs = [lead]
    return [s.strip(" ,;") for s in segs if s.strip(" ,;")]


def parse_page(page: str, article: int, day: date | None = None) -> list[Award]:
    """Every award on one announcement page."""
    day = day or title_date(page)
    if day is None:
        return []
    out: list[Award] = []
    branch = ""
    paras = paragraphs(page)
    i = 0
    while i < len(paras):
        p = paras[i]
        if is_heading(p):
            branch = re.sub(r"^Contracts\s+", "", p.title() if p.isupper() else p)
            i += 1
            continue
        if p.upper().startswith(("CORRECTION", "UPDATE")) or p.lstrip("* ").lower().startswith(
            ("small business", "woman-owned", "women-owned", "service-disabled", "hubzone")
        ):
            i += 1
            continue
        verb = _VERB.search(p)
        if verb is None:
            i += 1
            continue
        lead, rest = p[: verb.start()], p[verb.start() :]
        text = p
        listed: list[str] = []
        if _FOLLOWING.match(lead) or re.search(r"\bthe following\b", rest[:200], re.I):
            j = i + 1
            while j < len(paras) and not is_heading(paras[j]) and not _VERB.search(paras[j]):
                listed.extend(split_awardees(paras[j]))
                j += 1
            if listed:
                text = p + " " + " ".join(paras[i + 1 : j])
                i = j - 1
        names = listed or split_awardees(lead)
        if _FOLLOWING.match(lead) and not listed:
            names = []
        plural = bool(_PLURAL.match(rest)) or len(names) > 1
        n = max(len(names), 2 if plural else 1)
        kind = classify(text, n, award_phrase(rest))
        value = money(rest)
        flags = []
        if value is None:
            flags.append("no_value")
        contract = piid(rest)
        mod_no = _MOD_NO.search(rest)
        for k, name_lead in enumerate(names or [lead]):
            name, city, state, small, own = split_place(name_lead)
            out.append(
                Award(
                    date=day,
                    article=article,
                    para=i,
                    branch=branch,
                    company=name,
                    city=city,
                    state=state,
                    small_business=small,
                    value=value,
                    contract=(own[0] if own and n > 1 else "")
                    or contract
                    or (own[0] if own else ""),
                    mod_number=mod_no.group(1) if mod_no else "",
                    type=kind,
                    group=GROUP[kind],
                    ceiling=bool(_CEILING.search(rest)),
                    n_awardees=n,
                    text=text[:4000],
                    awardee_index=k,
                    flags=list(flags),
                )
            )
        i += 1
    return out


def frame(awards: list[Award]) -> pd.DataFrame:
    cols = list(Award.__dataclass_fields__)
    rows: list[dict[str, Any]] = [asdict(a) for a in awards]
    df = pd.DataFrame(rows, columns=cols)
    df["flags"] = df["flags"].map(lambda f: ",".join(f) if isinstance(f, list) else str(f))
    return df
