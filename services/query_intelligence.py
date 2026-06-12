"""
Query Intelligence Layer
Transforms raw user input into optimized multi-source search queries.
Handles: intent detection, typo correction, synonym expansion,
abbreviation expansion, entity extraction, and query diversification.
"""

import re
from dataclasses import dataclass, field

# ── Domain synonym map ─────────────────────────────────────────────────────
SYNONYMS: dict[str, list[str]] = {
    # Fibre types
    "cfrp":          ["carbon fibre reinforced polymer", "carbon fiber composite", "carbon fibre composite"],
    "gfrp":          ["glass fibre reinforced polymer", "glass fiber composite", "fibreglass composite"],
    "nfrp":          ["natural fibre reinforced polymer", "natural fiber composite", "bio-composite"],
    "cf":            ["carbon fibre", "carbon fiber"],
    "gf":            ["glass fibre", "glass fiber"],
    "nf":            ["natural fibre", "natural fiber"],

    # Natural fibres
    "jute":          ["jute fibre", "corchorus", "bast fibre"],
    "flax":          ["flax fibre", "linum usitatissimum", "linseed fibre", "bast fibre"],
    "hemp":          ["hemp fibre", "cannabis sativa", "bast fibre"],
    "kenaf":         ["kenaf fibre", "hibiscus cannabinus"],
    "sisal":         ["sisal fibre", "agave sisalana"],
    "ramie":         ["ramie fibre", "china grass"],
    "coir":          ["coir fibre", "coconut fibre"],
    "bamboo":        ["bamboo fibre", "bamboo composite"],
    "basalt":        ["basalt fibre", "basalt fiber", "volcanic fiber"],

    # Manufacturing
    "rtm":           ["resin transfer moulding", "resin transfer molding", "liquid composite moulding"],
    "vartm":         ["vacuum assisted resin transfer moulding", "VARI", "vacuum infusion"],
    "prepreg":       ["pre-impregnated composite", "preimpregnated laminate"],
    "hand layup":    ["hand lay-up", "manual layup", "wet layup"],
    "autoclave":     ["autoclave curing", "pressure curing composites"],
    "3d printing":   ["additive manufacturing composites", "FDM composites", "3D printed composite"],
    "braiding":      ["braided composite", "3D braiding", "braid architecture"],
    "weaving":       ["woven composite", "woven fabric", "woven reinforcement", "woven preform"],

    # Mechanical tests
    "tensile":       ["tensile strength", "tensile test", "uniaxial tension", "stress-strain"],
    "flexural":      ["flexural strength", "bending strength", "3-point bend", "four-point bend"],
    "impact":        ["impact strength", "charpy impact", "izod impact", "drop weight impact"],
    "fatigue":       ["fatigue life", "cyclic loading", "S-N curve", "fatigue damage"],
    "creep":         ["creep behaviour", "time-dependent deformation", "viscoelastic"],
    "shear":         ["shear strength", "ILSS", "interlaminar shear", "in-plane shear"],
    "compression":   ["compressive strength", "compression test", "buckling"],
    "hardness":      ["vickers hardness", "rockwell hardness", "shore hardness", "indentation"],
    "fracture":      ["fracture toughness", "crack propagation", "fracture mechanics", "KIC"],
    "delamination":  ["interlaminar failure", "ply separation", "mode I fracture", "mode II fracture"],

    # Properties / characterisation
    "fvf":           ["fibre volume fraction", "fiber volume fraction", "reinforcement content"],
    "voidcontent":   ["void content", "porosity", "voids", "defects"],
    "density":       ["specific gravity", "weight", "mass per unit volume"],
    "moisture":      ["water absorption", "moisture uptake", "hygroscopic"],
    "thermal":       ["thermal conductivity", "thermal stability", "TGA", "DSC", "heat resistance"],
    "electrical":    ["electrical conductivity", "resistivity", "EMI shielding"],
    "shm":           ["structural health monitoring", "damage detection", "strain sensing"],
    "ndt":           ["non-destructive testing", "ultrasonic testing", "CT scan composites", "X-ray composites"],

    # Treatments
    "alkali":        ["NaOH treatment", "alkali treatment", "mercerization", "chemical treatment"],
    "silane":        ["silane coupling agent", "surface treatment", "surface modification"],
    "surface treat": ["surface modification", "fibre treatment", "compatibilization"],

    # Modelling / simulation
    "fem":           ["finite element method", "finite element analysis", "FEA", "ABAQUS", "ANSYS"],
    "ruc":           ["representative unit cell", "unit cell model", "mesoscale model"],
    "clt":           ["classical laminate theory", "laminate analysis"],
    "md":            ["molecular dynamics", "atomistic simulation"],
    "ml":            ["machine learning composites", "neural network composites", "deep learning composites"],
    "digital twin":  ["digital twin composites", "virtual model", "physics-based model"],
    "monte carlo":   ["probabilistic analysis", "stochastic simulation", "uncertainty quantification"],

    # Aerospace terms
    "damage tolerance": ["damage tolerant design", "residual strength", "notch sensitivity"],
    "cai":           ["compression after impact", "post-impact compression"],
    "tai":           ["tension after impact"],
    "bvid":          ["barely visible impact damage", "impact damage threshold"],
    "glare":         ["glass laminate aluminium reinforced epoxy", "fibre metal laminate", "FML"],
    "sandwich":      ["sandwich structure", "sandwich panel", "core material", "foam core", "honeycomb"],

    # Textile terms
    "preform":       ["fibre preform", "fabric preform", "near net shape", "dry fabric"],
    "ncf":           ["non-crimp fabric", "stitched fabric", "multiaxial fabric"],
    "tow":           ["fibre tow", "yarn bundle", "fibre bundle"],
    "weave pattern": ["plain weave", "twill weave", "satin weave", "harness weave"],
}

# ── Abbreviation expander ──────────────────────────────────────────────────
ABBREVIATIONS: dict[str, str] = {
    "cfrp": "carbon fibre reinforced polymer",
    "gfrp": "glass fibre reinforced polymer",
    "nfrp": "natural fibre reinforced polymer",
    "fvf":  "fibre volume fraction",
    "rtm":  "resin transfer moulding",
    "fem":  "finite element method",
    "fea":  "finite element analysis",
    "shm":  "structural health monitoring",
    "ndt":  "non-destructive testing",
    "clt":  "classical laminate theory",
    "ruc":  "representative unit cell",
    "ncf":  "non-crimp fabric",
    "cai":  "compression after impact",
    "bvid": "barely visible impact damage",
    "glare":"glass laminate aluminium reinforced epoxy",
    "fml":  "fibre metal laminate",
    "ilss": "interlaminar shear strength",
    "vartm":"vacuum assisted resin transfer moulding",
    "ep":   "epoxy",
    "pp":   "polypropylene",
    "pa":   "polyamide",
    "peek": "polyetheretherketone",
    "pps":  "polyphenylene sulphide",
    "tga":  "thermogravimetric analysis",
    "dsc":  "differential scanning calorimetry",
    "sem":  "scanning electron microscopy",
    "tem":  "transmission electron microscopy",
    "xrd":  "X-ray diffraction",
    "afm":  "atomic force microscopy",
    "dic":  "digital image correlation",
    "ae":   "acoustic emission",
    "uts":  "ultimate tensile strength",
    "ys":   "yield strength",
    "e":    "Young's modulus",
    "3dwc": "3D woven composite",
    "3dbc": "3D braided composite",
}

# ── Common typo patterns ───────────────────────────────────────────────────
TYPO_CORRECTIONS: dict[str, str] = {
    "composit":      "composite",
    "compsite":      "composite",
    "compoiste":     "composite",
    "fibre":         "fibre",   # already correct (British)
    "fiber":         "fibre",   # normalise to British
    "fibour":        "fibre",
    "laminat":       "laminate",
    "lamnate":       "laminate",
    "tenslie":       "tensile",
    "flexrual":      "flexural",
    "mecanical":     "mechanical",
    "mechanicl":     "mechanical",
    "properites":    "properties",
    "properteis":    "properties",
    "strenght":      "strength",
    "stregth":       "strength",
    "reinfoced":     "reinforced",
    "reinfored":     "reinforced",
    "aeorspace":     "aerospace",
    "aerosapce":     "aerospace",
    "texile":        "textile",
    "textlie":       "textile",
    "polyemer":      "polymer",
    "polimer":       "polymer",
    "epoxy":         "epoxy",
    "hybrd":         "hybrid",
    "hybird":        "hybrid",
    "wovne":         "woven",
    "woevn":         "woven",
    "braied":        "braided",
    "braieded":      "braided",
    "alkalie":       "alkali",
    "treatement":    "treatment",
    "naoh":          "NaOH",
    "defomration":   "deformation",
    "deforamtion":   "deformation",
}

# ── Intent patterns ────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "property_lookup": [
        r"\b(what is|find|get|show).*(strength|modulus|stiffness|density|conductivity)\b",
        r"\b(tensile|flexural|impact|shear|compression).*(strength|modulus|test)\b",
        r"\b(mechanical|physical|thermal|electrical)\s+propert",
        r"\b(hybrid|natural).*(composite|laminate|fibre|fiber)\b",
        r"\b(jute|flax|hemp|sisal|kenaf).*(composite|fibre|hybrid)\b",
        r"\b(composite|laminate).*(propert|characteris|behav)\b",
    ],
    "fabrication": [
        r"\b(how to|process|fabricat|manufactur|mak|produc|creat)\b",
        r"\b(rtm|vartm|hand lay|autoclave|prepreg|infusion)\b",
        r"\b(cure|mould|mold|laminate|stack|layup)\b",
    ],
    "review": [
        r"\b(review|overview|survey|state of art|recent|progress|advance)\b",
        r"\b(compare|comparison|vs\.?|versus|between)\b",
    ],
    "modelling": [
        r"\b(model|simulat|fem|fea|predict|analys|numerica)\b",
        r"\b(finite element|unit cell|damage model|progressive failure)\b",
    ],
    "treatment": [
        r"\b(treat|modif|surface|alkali|silane|chemical|naoh)\b",
        r"\b(compatibil|adhesion|interface|bonding)\b",
    ],
    "characterisation": [
        r"\b(characteris|sem|tem|xrd|ftir|ndt|ct scan|ultrasonic|x-ray)\b",
        r"\b(microstructure|morpholog|defect|void|porosity)\b",
    ],
}

# ── Discipline keyword triggers ────────────────────────────────────────────
DISCIPLINE_TRIGGERS = {
    "aerospace": [
        "aerospace", "aircraft", "airframe", "fuselage", "wing", "spacecraft",
        "aerostructure", "damage tolerance", "cai", "bvid", "glare", "fml",
        "airbus", "boeing", "aeronautics", "uav", "drone structure",
    ],
    "materials": [
        "composite", "laminate", "epoxy", "matrix", "reinforced", "hybrid",
        "nanocomposite", "fracture", "delamination", "interfacial", "curing",
        "void", "porosity", "mechanical properties", "microstructure",
    ],
    "textile": [
        "woven", "braided", "knitted", "nonwoven", "preform", "fabric",
        "yarn", "tow", "jute", "flax", "hemp", "sisal", "kenaf", "ramie",
        "coir", "basalt", "natural fibre", "ncf", "weave", "textile",
    ],
}


@dataclass
class QueryAnalysis:
    original:           str
    corrected:          str
    intent:             str
    discipline:         str
    entities:           list[str]     = field(default_factory=list)
    expanded_terms:     list[str]     = field(default_factory=list)
    primary_query:      str           = ""
    secondary_queries:  list[str]     = field(default_factory=list)
    year_hint:          int | None    = None
    is_review_request:  bool          = False


def _correct_typos(text: str) -> str:
    words = text.lower().split()
    corrected = []
    for word in words:
        clean = re.sub(r'[^a-z0-9]', '', word)
        corrected.append(TYPO_CORRECTIONS.get(clean, word))
    return " ".join(corrected)


def _expand_abbreviations(text: str) -> str:
    words = text.lower().split()
    expanded = []
    for word in words:
        clean = re.sub(r'[^a-z0-9]', '', word)
        if clean in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[clean])
        else:
            expanded.append(word)
    return " ".join(expanded)


def _detect_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return intent
    return "general"


def _detect_discipline(text: str, requested_discipline: str) -> str:
    if requested_discipline != "all":
        return requested_discipline
    text_lower = text.lower()
    scores = {d: 0 for d in DISCIPLINE_TRIGGERS}
    for disc, triggers in DISCIPLINE_TRIGGERS.items():
        for t in triggers:
            if t in text_lower:
                scores[disc] += 1
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "all"


def _extract_entities(text: str) -> list[str]:
    entities = []
    text_lower = text.lower()
    entity_patterns = [
        r'\b(jute|flax|hemp|kenaf|sisal|ramie|coir|bamboo|basalt)\b',
        r'\b(carbon\s+fi(?:bre|ber)|glass\s+fi(?:bre|ber)|aramid|kevlar)\b',
        r'\b(epoxy|polyester|vinylester|polyurethane|peek|pps|pp|pa6)\b',
        r'\b(woven|braided|knitted|nonwoven|ncf|prepreg|3d\s+woven)\b',
        r'\b(\d+(?:\.\d+)?%?\s*(?:fvf|fibre volume|fiber volume))\b',
        r'\b(tensile|flexural|impact|shear|compressive|fatigue)\b',
        r'\b(alkali|naoh|silane|surface\s+treat)\b',
        r'\b(fem|fea|clt|monte carlo|machine learning)\b',
        r'\b(\d{4})\b',  # years
    ]
    for pat in entity_patterns:
        matches = re.findall(pat, text_lower)
        entities.extend(m if isinstance(m, str) else m[0] for m in matches if m)
    return list(dict.fromkeys(entities))  # deduplicate preserving order


def _expand_query_terms(text: str) -> list[str]:
    """Find synonyms for terms in the query."""
    text_lower = text.lower()
    expansions = []
    for key, synonyms in SYNONYMS.items():
        if key in text_lower:
            expansions.extend(synonyms[:2])  # max 2 synonyms per term
    return list(dict.fromkeys(expansions))


def _build_queries(analysis: 'QueryAnalysis') -> tuple[str, list[str]]:
    """
    Build primary query and 2-3 diverse secondary queries.
    Primary: concise, high-signal query.
    Secondary: broader, narrower, and alternate-phrasing versions.
    """
    base = analysis.corrected

    # Expand abbreviations in base
    base_expanded = _expand_abbreviations(base)

    # Primary query = corrected + key expanded terms (max 8 words)
    primary_terms = base_expanded.split()[:8]
    primary = " ".join(primary_terms)

    # Secondary 1: add top synonym expansions
    sec1_terms = list(dict.fromkeys(primary.split() + analysis.expanded_terms[:4]))
    sec1 = " ".join(sec1_terms[:10])

    # Secondary 2: intent-focused variant
    intent_boosts = {
        "property_lookup": "mechanical properties characterisation testing",
        "fabrication":     "manufacturing process fabrication method",
        "review":          "review recent advances overview",
        "modelling":       "numerical model finite element simulation",
        "treatment":       "surface treatment modification interface",
        "characterisation":"microstructure morphology defect analysis",
        "general":         "",
    }
    boost = intent_boosts.get(analysis.intent, "")
    sec2 = f"{primary} {boost}".strip() if boost else ""

    # Secondary 3: entity-rich query
    key_entities = analysis.entities[:5]
    if key_entities:
        sec3 = " ".join(key_entities)
    else:
        sec3 = ""

    secondaries = [q for q in [sec1, sec2, sec3] if q and q != primary]
    return primary, secondaries


def analyse_query(raw_query: str, discipline: str = "all") -> QueryAnalysis:
    """
    Full query intelligence pipeline.
    Returns a QueryAnalysis with everything needed for smart search.
    """
    # Step 1: correct typos
    corrected = _correct_typos(raw_query.strip())

    # Step 2: detect intent
    intent = _detect_intent(corrected)

    # Step 3: auto-detect discipline if not specified
    detected_discipline = _detect_discipline(corrected, discipline)

    # Step 4: extract entities
    entities = _extract_entities(corrected)

    # Step 5: expand synonyms
    expanded = _expand_query_terms(corrected)

    # Step 6: extract year hint
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', corrected)
    year_hint  = int(year_match.group()) if year_match else None

    analysis = QueryAnalysis(
        original=raw_query,
        corrected=corrected,
        intent=intent,
        discipline=detected_discipline,
        entities=entities,
        expanded_terms=expanded,
        year_hint=year_hint,
        is_review_request="review" in intent or bool(re.search(r'\breview\b', corrected)),
    )

    # Step 7: build optimised queries
    primary, secondaries = _build_queries(analysis)
    analysis.primary_query     = primary
    analysis.secondary_queries = secondaries

    return analysis


def rerank_results(papers: list[dict], analysis: QueryAnalysis) -> list[dict]:
    """
    Rerank papers by intent relevance, not just source ordering.
    Scoring factors:
      - Title/abstract keyword match with expanded terms
      - Entity match (specific fibres, processes, tests mentioned)
      - Citation count (proxy for importance)
      - Year (recency bonus for non-review queries)
      - Discipline alignment
    """
    import time

    CURRENT_YEAR = 2026

    def score(paper: dict) -> float:
        title    = (paper.get("title")    or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        text     = title + " " + abstract

        s = 0.0

        # ── Corrected query terms match ──
        for term in analysis.corrected.lower().split():
            if len(term) > 3:
                if term in title:    s += 3.0
                if term in abstract: s += 1.0

        # ── Expanded synonym terms ──
        for term in analysis.expanded_terms:
            if term.lower() in text: s += 1.5

        # ── Entity match (high signal) ──
        for entity in analysis.entities:
            if entity.lower() in title:    s += 4.0
            if entity.lower() in abstract: s += 2.0

        # ── Citation count (log scale) ──
        citations = paper.get("citation_count") or 0
        if citations > 0:
            import math
            s += math.log(citations + 1) * 0.8

        # ── Recency (unless review request) ──
        year = paper.get("year")
        if year and not analysis.is_review_request:
            age = CURRENT_YEAR - year
            if age <= 3:   s += 3.0
            elif age <= 6: s += 1.5
            elif age <= 10: s += 0.5

        # ── Discipline alignment ──
        disc_tag = paper.get("discipline_tag", "general")
        if analysis.discipline != "all" and disc_tag == analysis.discipline:
            s += 3.0

        # ── Abstract presence bonus ──
        if paper.get("abstract"):
            s += 1.0

        # ── Open access bonus ──
        if paper.get("open_access_url"):
            s += 0.5

        return s

    scored = [(paper, score(paper)) for paper in papers]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored]
