"""
Paper quality scoring with weighted formula:
  score = 0.40 * relevance
        + 0.25 * citation_score
        + 0.15 * recency_score
        + 0.10 * journal_quality
        + 0.10 * open_access_bonus

Material-type guard: papers that don't mention the discipline's core
materials are penalised heavily so they don't crowd out on-topic results.
"""
import math
import re

CURRENT_YEAR = 2026

# ── Journal tiers ──────────────────────────────────────────────────────────
TIER1_JOURNALS = {
    "composites science and technology",
    "composites part a",
    "composites part b",
    "composite structures",
    "journal of composite materials",
    "carbon",
    "polymer composites",
    "textile research journal",
    "journal of the textile institute",
    "aiaa journal",
    "aerospace science and technology",
    "journal of aerospace engineering",
    "acta astronautica",
    "international journal of mechanical sciences",
    "materials & design",
    "engineering fracture mechanics",
    "international journal of fatigue",
}

TIER2_JOURNALS = {
    "materials today",
    "polymers",
    "fibers and polymers",
    "journal of materials science",
    "applied composite materials",
    "journal of reinforced plastics and composites",
    "industrial crops and products",
    "construction and building materials",
    "thin-walled structures",
    "journal of natural fibres",
    "cellulose",
    "carbohydrate polymers",
    "bioresource technology",
    "heliyon",
    "materials",
}

# ── Discipline material-type vocabularies ──────────────────────────────────
# A paper matching the discipline filter must mention at least one term
# from the relevant set (title OR abstract). If it matches zero, it gets a
# score multiplier < 1.0 — enough to push it behind on-topic results.
DISCIPLINE_MATERIAL_TERMS: dict[str, list[str]] = {
    "aerospace": [
        # polymer matrix composites
        "cfrp", "carbon fibre", "carbon fiber", "carbon/epoxy",
        "carbon fibre reinforced", "carbon fiber reinforced",
        "fibre reinforced polymer", "fiber reinforced polymer",
        "frp", "polymer matrix composite", "epoxy composite",
        "glass fibre reinforced", "glass fiber reinforced",
        "gfrp", "kevlar", "aramid", "polymer composite",
        # aerospace-specific
        "composite laminate", "composite structure", "aerostructure",
        "airframe", "fuselage", "wing spar", "sandwich panel",
        "damage tolerance", "delamination", "interlaminar",
        "structural health monitoring", "shm",
        "prepreg", "autoclave", "out-of-autoclave",
        # aluminium alloys are legitimate aerospace (not penalised)
        "aluminium alloy", "aluminum alloy", "al alloy",
        "aa2024", "aa7075", "titanium alloy", "ti-6al",
        # generic fatigue in composite context
        "fatigue crack growth composite", "composite fatigue",
    ],
    "materials": [
        "composite", "polymer", "epoxy", "fibre reinforced", "fiber reinforced",
        "matrix", "laminate", "nanocomposite", "hybrid composite",
        "tensile strength", "flexural strength", "young's modulus",
        "fracture toughness", "delamination", "void content",
        "fibre volume", "fiber volume", "resin", "curing",
        "metal matrix", "ceramic matrix", "carbon nanotube",
        "graphene", "nanoparticle",
    ],
    "textile": [
        "jute", "flax", "hemp", "ramie", "kenaf", "sisal", "coir",
        "bamboo", "natural fibre", "natural fiber", "bast fibre", "bast fiber",
        "bio-composite", "biocomposite", "woven", "nonwoven", "fabric",
        "textile composite", "technical textile", "braided", "woven reinforcement",
        "preform", "natural reinforcement", "plant fibre", "plant fiber",
        "cellulose fibre", "cellulose fiber", "pla composite", "biopolymer",
    ],
}

# Off-topic material signals — if these appear prominently WITHOUT any
# discipline term, the paper is almost certainly off-topic.
OFF_TOPIC_SIGNALS: dict[str, list[str]] = {
    "aerospace": [
        "bearing steel", "hydrogen embrittlement", "reinforced concrete",
        "concrete beam", "soil", "wood", "timber", "rock", "geological",
        "biological", "soft tissue", "bone", "hydrogel",
        "aluminium matrix composite",   # MMC, not CFRP
        "short-fibre reinforced aluminium",
        "reinforced aluminium",
    ],
    "materials": [
        "reinforced concrete", "soil stabilisation", "geological",
        "biological tissue", "soft tissue", "bone fracture",
    ],
    "textile": [
        "reinforced concrete", "steel fibre concrete", "steel fiber concrete",
        "bearing steel", "aluminium alloy", "titanium alloy",
    ],
}


def _material_match_multiplier(paper: dict, discipline: str) -> float:
    """
    Returns a score multiplier [0.15 … 1.0] based on whether the paper's
    text contains material terms appropriate for the searched discipline.

    1.0  — contains ≥1 on-topic material term  (no penalty)
    0.40 — zero on-topic terms but no explicit off-topic signal
    0.15 — contains an explicit off-topic signal
    """
    if discipline not in DISCIPLINE_MATERIAL_TERMS:
        return 1.0   # "all" discipline — no filter

    title    = (paper.get("title")    or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    text     = title + " " + abstract

    on_topic_terms  = DISCIPLINE_MATERIAL_TERMS[discipline]
    off_topic_terms = OFF_TOPIC_SIGNALS.get(discipline, [])

    has_on_topic  = any(t in text for t in on_topic_terms)
    has_off_topic = any(t in text for t in off_topic_terms)

    if has_off_topic:
        return 0.15   # very strong penalty — push to bottom
    if has_on_topic:
        return 1.0    # on-topic — no penalty
    return 0.40       # no signal either way — mild penalty


# ── Scoring components ─────────────────────────────────────────────────────

def _relevance_score(paper: dict, query_terms: list[str], entities: list[str]) -> float:
    """Score 0-1 based on term/entity match in title + abstract."""
    title    = (paper.get("title")    or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    text     = title + " " + abstract

    if not text.strip():
        return 0.1

    total_terms = len(query_terms) + len(entities)
    if total_terms == 0:
        return 0.5

    score = 0.0
    for term in query_terms:
        if len(term) > 2:
            if term in title:       score += 2.0
            elif term in abstract:  score += 0.8

    for entity in entities:
        if entity in title:         score += 3.0
        elif entity in abstract:    score += 1.5

    max_possible = len(query_terms) * 2 + len(entities) * 3
    return min(1.0, score / max(max_possible, 1))


def _citation_score(citation_count: int) -> float:
    """Log-normalised citation score 0-1. 100 citations ≈ 0.85."""
    if citation_count <= 0:
        return 0.0
    return min(1.0, math.log(citation_count + 1) / math.log(101))


def _recency_score(year: int | None) -> float:
    """Recency score 0-1. Recent papers score higher."""
    if not year:
        return 0.3
    age = CURRENT_YEAR - year
    if age <= 2:   return 1.0
    if age <= 4:   return 0.85
    if age <= 7:   return 0.65
    if age <= 12:  return 0.40
    if age <= 20:  return 0.20
    return 0.10


def _journal_quality_score(journal: str | None) -> float:
    """Journal tier score 0-1."""
    if not journal:
        return 0.3
    j = journal.lower()
    for t1 in TIER1_JOURNALS:
        if t1 in j:
            return 1.0
    for t2 in TIER2_JOURNALS:
        if t2 in j:
            return 0.65
    if "arxiv" in j:
        return 0.35   # preprint
    return 0.40


def _oa_score(open_access_url: str | None) -> float:
    return 1.0 if open_access_url else 0.0


# ── Public API ─────────────────────────────────────────────────────────────

def score_paper(
    paper: dict,
    query_terms: list[str],
    entities: list[str],
    discipline: str = "all",
) -> float:
    """
    Compute weighted quality score for a paper, with discipline-aware
    material-type penalty applied as a final multiplier.

    Weights:
      0.40 relevance       — keyword/entity match in title + abstract
      0.25 citation        — log-normalised citation count
      0.15 recency         — publication year
      0.10 journal quality — tier-1 / tier-2 / unknown / preprint
      0.10 open access     — has free PDF
    × material multiplier  — 1.0 on-topic / 0.40 neutral / 0.15 off-topic
    """
    r  = _relevance_score(paper, query_terms, entities)
    c  = _citation_score(paper.get("citation_count") or 0)
    t  = _recency_score(paper.get("year"))
    j  = _journal_quality_score(paper.get("journal"))
    oa = _oa_score(paper.get("open_access_url"))

    base  = (0.40 * r) + (0.25 * c) + (0.15 * t) + (0.10 * j) + (0.10 * oa)
    multi = _material_match_multiplier(paper, discipline)
    return round(base * multi, 4)


def rank_papers(
    papers: list[dict],
    query_terms: list[str],
    entities: list[str],
    discipline: str = "all",
) -> list[dict]:
    """Score and sort papers by quality score descending."""
    for p in papers:
        p["_quality_score"] = score_paper(p, query_terms, entities, discipline)
    papers.sort(key=lambda x: x["_quality_score"], reverse=True)
    return papers
