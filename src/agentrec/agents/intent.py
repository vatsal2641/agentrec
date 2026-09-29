"""Structured request constraints + a regex parser.

`Constraints` is the *structured output* the agent works with. A real LLM is
asked to produce it (via tool arguments); the regex parser below is the
deterministic stand-in used for tests/offline evaluation. The regex parser is
brittle by design — it is exactly the component an LLM should beat on messy,
open-ended language ("something cosy for a rainy Sunday, nothing too long").
"""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field

GENRE_SYNONYMS: dict[str, list[str]] = {
    "Action": ["action"], "Adventure": ["adventure"], "Animation": ["animation", "animated", "cartoon"],
    "Children's": ["children", "children's", "kids", "family"], "Comedy": ["comedy", "comedies", "funny"],
    "Crime": ["crime", "gangster"], "Documentary": ["documentary", "documentaries"], "Drama": ["drama", "dramas"],
    "Fantasy": ["fantasy"], "Film-Noir": ["noir", "film-noir"], "Horror": ["horror", "scary"],
    "Musical": ["musical", "musicals"], "Mystery": ["mystery", "mysteries"],
    "Romance": ["romance", "romantic", "romcom"], "Sci-Fi": ["sci-fi", "scifi", "science fiction", "sci fi"],
    "Thriller": ["thriller", "thrillers", "suspense"], "War": ["war"], "Western": ["western", "westerns"],
}
_NEG = r"\b(?:no|not|without|avoid|except|nothing|never|hate|don't like|do not like|dislike|skip)\b"


@dataclass
class Constraints:
    include_genres: list[str] = field(default_factory=list)   # soft: at least one should match
    exclude_genres: list[str] = field(default_factory=list)   # HARD: never violated, never relaxed
    year_min: int | None = None                               # soft
    year_max: int | None = None                               # soft
    similar_to: str | None = None                             # a title the user referenced
    popularity: str | None = None                             # "head" | "tail" | None (soft)
    k: int = 5
    persistent_excludes: list[str] = field(default_factory=list)  # "I hate horror" -> remember

    def to_dict(self) -> dict:
        return asdict(self)


def _find_genres(text: str) -> list[tuple[str, int]]:
    out = []
    for g, syns in GENRE_SYNONYMS.items():
        for s in syns:
            for m in re.finditer(rf"\b{re.escape(s)}\b", text):
                out.append((g, m.start()))
    return sorted(out, key=lambda x: x[1])   # textual order


def parse_request(text: str, titles: list[str] | None = None, max_year: int = 2000) -> Constraints:
    t = text.lower()
    c = Constraints()
    if m := re.search(r"\b(\d{1,2})\s+(?:[a-z'-]+\s+){0,3}(?:movies|films|recommendations|picks|options)\b", t):
        c.k = max(1, min(20, int(m.group(1))))
    elif m := re.search(r"\btop\s+(\d{1,2})\b", t):
        c.k = max(1, min(20, int(m.group(1))))

    # similar_to: text after "like" / "similar to" — quoted, or up to a clause break
    if m := re.search(r"(?:similar to|like)\s+[\"“']([^\"”']+)[\"”']", text, flags=re.I):
        c.similar_to = m.group(1).strip()
    elif m := re.search(r"(?:similar to|something like|movies like|films like)\s+(.+?)(?:\s+but\b|,|\.|$)", text, flags=re.I):
        c.similar_to = m.group(1).strip()
    if c.similar_to and titles:
        c.similar_to = match_title(c.similar_to, titles) or c.similar_to

    sim_span = t.find(c.similar_to.lower()) if c.similar_to else -1
    for g, pos in _find_genres(t):
        if c.similar_to and sim_span <= pos < sim_span + len(c.similar_to):
            continue                                    # genre word inside a title, ignore
        window = t[max(0, pos - 25):pos]
        if re.search(_NEG + r"[\w\s']{0,15}$", window):
            if g not in c.exclude_genres:
                c.exclude_genres.append(g)
            if re.search(r"\b(hate|never|don't like|do not like|dislike)\b", window) and g not in c.persistent_excludes:
                c.persistent_excludes.append(g)
        elif g not in c.include_genres:
            c.include_genres.append(g)
    c.include_genres = [g for g in c.include_genres if g not in c.exclude_genres]

    if m := re.search(r"\b(?:19)?(\d)0'?s\b", t):
        dec = 1900 + int(m.group(1)) * 10
        c.year_min, c.year_max = dec, dec + 9
    if m := re.search(r"\b(?:after|since|newer than)\s+(\d{4})", t):
        c.year_min = int(m.group(1))
    if m := re.search(r"\b(?:before|older than)\s+(\d{4})", t):
        c.year_max = int(m.group(1))
    if re.search(r"\b(classic|old)\b", t) and c.year_max is None:
        c.year_max = 1979
    if re.search(r"\b(recent|new)\b", t) and c.year_min is None:
        c.year_min = max_year - 5
    if re.search(r"hidden gem|underrated|lesser[- ]known|obscure", t):
        c.popularity = "tail"
    elif re.search(r"\b(popular|blockbuster|crowd[- ]pleaser)", t):
        c.popularity = "head"
    return c


def match_title(query: str, titles: list[str], cutoff: float = 0.6) -> str | None:
    """Fuzzy title match (ignores the '(year)' suffix)."""
    base = {re.sub(r"\s*\(\d{4}\)\s*$", "", t).lower(): t for t in titles}
    q = re.sub(r"\s*\(\d{4}\)\s*$", "", query).lower().strip()
    if q in base:
        return base[q]
    best = difflib.get_close_matches(q, list(base), n=1, cutoff=cutoff)
    return base[best[0]] if best else None
