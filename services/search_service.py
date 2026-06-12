"""
Multi-source academic search with:
- LLM query rewriting
- 4 parallel sources: OpenAlex · Crossref · arXiv · PubMed
- Deduplication
- Weighted paper quality scoring
- Unpaywall OA enrichment
"""
import httpx
import asyncio
import hashlib
import re
import time
import logging
import xml.etree.ElementTree as ET

from models.schemas import Paper, Author
from services.paper_scoring import rank_papers
from services.query_intelligence import analyse_query
from agents.query_agent import rewrite_query

logger = logging.getLogger(__name__)

# ── Cache ──────────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}
CACHE_TTL   = 3600
CACHE_MAXSIZE = 256   # BUG FIX: cap to prevent unbounded memory growth

DISCIPLINE_CONCEPTS = {
    "aerospace": "C27206212",
    "materials": "C192562407",
    "textile":   "C107038049",
}

DISCIPLINE_KEYWORDS = {
    "aerospace": ["aerospace","aeronautics","aircraft","spacecraft","cfrp","carbon fibre","carbon fiber","airframe","aerostructure","damage tolerance","composite laminate","sandwich panel","fatigue crack","structural health monitoring"],
    "materials": ["composite","hybrid composite","nanocomposite","polymer matrix","epoxy","fibre reinforced","fiber reinforced","tensile strength","flexural strength","void content","fibre volume fraction","delamination","fracture toughness","matrix cracking","interlaminar"],
    "textile":   ["textile composite","woven composite","natural fibre composite","natural fiber composite","jute composite","flax composite","hemp composite","jute/flax","flax/jute","jute/glass","flax/glass","jute","flax","hemp","ramie","kenaf","sisal","coir","bast fibre","bast fiber","preform","woven reinforcement","technical textile","braided composite","bio-composite","pla composite"],
}

EXCLUDE_CONCEPTS = {"C127413603","C144024400","C185592680"}

JUNK_TITLE_PATTERNS = [r"^specification for",r"^standard for",r"^iso \d+",r"^bs \d+",r"^astm [a-z]",r"twines made from",r"^patent"]
JUNK_JOURNALS       = ["revista canaria","english studies","literary","social science","economics","psychology","nursing","law review"]
NON_ENG_WORDS       = ["novel","poetry","poem","fiction","literature","artistry","pamela","shakespeare","biblical","rhetoric","narrative"]


# Natural fibre names — title match forces textile tag
_NATURAL_FIBRES = {
    "jute","flax","hemp","ramie","kenaf","sisal","coir","bamboo",
    "jute/flax","flax/jute","jute/glass","flax/glass","jute/ramie",
    "natural fibre","natural fiber","bast fibre","bast fiber",
}

def _tag_discipline(title: str, abstract: str | None, concepts: list) -> str:
    title_lower = (title or "").lower()
    concept_names = " ".join(c.get("display_name","").lower() for c in concepts)
    text = (title_lower + " " + (abstract or "") + " " + concept_names).lower()

    # Strong signal: natural fibre names in title → textile
    for nf in _NATURAL_FIBRES:
        if nf in title_lower:
            return "textile"

    scores = {d: sum(1 for kw in kws if kw in text) for d, kws in DISCIPLINE_KEYWORDS.items()}
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


def _is_excluded(concepts: list) -> bool:
    return bool({c.get("id","").split("/")[-1] for c in concepts} & EXCLUDE_CONCEPTS)


def _is_junk(paper: dict) -> bool:
    title   = (paper.get("title")   or "").lower().strip()
    journal = (paper.get("journal") or "").lower()
    if not title: return True
    if not paper.get("authors") and not paper.get("year"): return True
    for pat in JUNK_TITLE_PATTERNS:
        if re.match(pat, title, re.IGNORECASE): return True
    for j in JUNK_JOURNALS:
        if j in journal: return True
    for w in NON_ENG_WORDS:
        if w in title: return True
    return False


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]","", (t or "").lower())[:80]


def _reconstruct_abstract(inv: dict | None) -> str | None:
    if not inv: return None
    try:
        pos = {}
        for word, positions in inv.items():
            for p in positions: pos[p] = word
        return " ".join(pos[i] for i in sorted(pos)) or None
    except: return None


def _clean_abstract(text: str | None) -> str | None:
    if not text: return None
    text = re.sub(r"<[^>]+>","",text).strip()
    return text if len(text) > 30 else None


def _cache_key(*args) -> str:
    return hashlib.md5("|".join(str(a) for a in args).encode()).hexdigest()


def _get_cached(key):
    e = _cache.get(key)
    if e is None:
        return None
    if time.time() >= e["expires"]:
        # BUG FIX: delete expired entries so the dict doesn't grow unboundedly
        del _cache[key]
        return None
    return e["papers"]


def _set_cache(key, papers):
    # BUG FIX: evict oldest entry when cache is full
    if len(_cache) >= CACHE_MAXSIZE:
        oldest_key = min(_cache, key=lambda k: _cache[k]["expires"])
        del _cache[oldest_key]
    _cache[key] = {"papers": papers, "expires": time.time() + CACHE_TTL}


# ── Abstract enrichment ────────────────────────────────────────────────────

async def _fetch_abstract_crossref(doi: str, client: httpx.AsyncClient) -> str | None:
    if not doi: return None
    doi_clean = doi.replace("https://doi.org/","").strip()
    try:
        resp = await client.get(f"https://api.crossref.org/works/{doi_clean}",timeout=7,headers={"User-Agent":"TriFieldAI/1.0 (contact@trifield.ai)"})
        if resp.status_code == 200:
            return _clean_abstract(resp.json().get("message",{}).get("abstract",""))
    except Exception as e:
        logger.debug(f"Crossref abstract fetch failed: {e}")
    return None


async def _fetch_abstract_s2(doi: str, client: httpx.AsyncClient) -> str | None:
    if not doi: return None
    doi_clean = doi.replace("https://doi.org/","").strip()
    try:
        resp = await client.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_clean}",params={"fields":"abstract"},timeout=7,headers={"User-Agent":"TriFieldAI/1.0"})
        if resp.status_code == 200:
            return _clean_abstract(resp.json().get("abstract",""))
    except Exception as e:
        logger.debug(f"S2 abstract fetch failed: {e}")
    return None


async def _get_oa_url(doi: str, client: httpx.AsyncClient) -> str | None:
    if not doi: return None
    doi_clean = doi.replace("https://doi.org/","").strip()
    try:
        resp = await client.get(f"https://api.unpaywall.org/v2/{doi_clean}?email=contact@trifield.ai",timeout=6)
        if resp.status_code == 200:
            best = resp.json().get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url")
    except Exception as e:
        logger.debug(f"Unpaywall fetch failed: {e}")
    return None


async def _enrich(paper: dict, client: httpx.AsyncClient) -> dict:
    doi = paper.get("doi") or ""
    abstract = paper.get("abstract")
    oa_url   = paper.get("open_access_url")
    if not abstract and doi:
        if not oa_url:
            abstract, oa_url = await asyncio.gather(_fetch_abstract_crossref(doi,client), _get_oa_url(doi,client))
        else:
            abstract = await _fetch_abstract_crossref(doi, client)
            if not abstract:
                abstract = await _fetch_abstract_s2(doi, client)
    elif not oa_url and doi:
        oa_url = await _get_oa_url(doi, client)
    if abstract:           paper["abstract"]        = abstract
    if oa_url:             paper["open_access_url"] = oa_url
    return paper


# ── Sources ────────────────────────────────────────────────────────────────

async def _search_openalex(query,discipline,year_from,year_to,limit,client):
    params={"search":query,"per_page":min(limit*2,50),"select":"id,title,authorships,publication_year,abstract_inverted_index,cited_by_count,doi,primary_location,open_access,concepts","sort":"relevance_score:desc","mailto":"contact@trifield.ai"}
    filters=[]
    if discipline!="all" and discipline in DISCIPLINE_CONCEPTS:filters.append(f"concepts.id:{DISCIPLINE_CONCEPTS[discipline]}")
    if year_from:filters.append(f"publication_year:>{year_from-1}")
    if year_to:filters.append(f"publication_year:<{year_to+1}")
    if filters:params["filter"]=",".join(filters)
    try:
        resp=await client.get("https://api.openalex.org/works",params=params,timeout=15)
        if resp.status_code!=200:return []
        papers=[]
        for item in resp.json().get("results",[]):
            doi=item.get("doi") or ""
            oa=item.get("open_access") or {}
            primary=item.get("primary_location") or {}
            source=(primary.get("source") or {})
            authors=[Author(name=a.get("author",{}).get("display_name","Unknown")) for a in item.get("authorships",[])[:10]]
            papers.append({"paper_id":item.get("id","").split("/")[-1],"title":item.get("title") or "","authors":authors,"year":item.get("publication_year"),"abstract":_reconstruct_abstract(item.get("abstract_inverted_index")),"citation_count":item.get("cited_by_count",0),"doi":doi,"url":doi or "","open_access_url":oa.get("oa_url"),"journal":source.get("display_name"),"concepts":item.get("concepts") or [],"source":"openalex"})
        return papers
    except Exception as e:
        logger.warning(f"OpenAlex search failed: {e}")
        return []


async def _search_crossref(query,discipline,year_from,year_to,limit,client):
    params={"query":query,"rows":min(limit*2,40),"select":"DOI,title,author,published,abstract,container-title,is-referenced-by-count,type","sort":"relevance","mailto":"contact@trifield.ai","filter":"type:journal-article,type:proceedings-article"}
    if year_from:params["filter"]+=f",from-pub-date:{year_from}"
    if year_to:params["filter"]+=f",until-pub-date:{year_to}"
    try:
        resp=await client.get("https://api.crossref.org/works",params=params,timeout=15,headers={"User-Agent":"TriFieldAI/1.0 (contact@trifield.ai)"})
        if resp.status_code!=200:return []
        papers=[]
        for item in resp.json().get("message",{}).get("items",[]):
            doi=item.get("DOI") or ""
            title_list=item.get("title") or []
            title=title_list[0] if title_list else ""
            if not title:continue
            authors=[Author(name=f"{a.get('given','')} {a.get('family','')}".strip()) for a in item.get("author",[])[:10]]
            pub=item.get("published") or {}
            date_parts=pub.get("date-parts") or [[None]]
            year=date_parts[0][0] if date_parts and date_parts[0] else None
            journal=(item.get("container-title") or [""])[0] or None
            papers.append({"paper_id":f"CR_{doi.replace('/','_')}","title":title,"authors":authors,"year":year,"abstract":_clean_abstract(item.get("abstract") or ""),"citation_count":item.get("is-referenced-by-count") or 0,"doi":f"https://doi.org/{doi}" if doi else "","url":f"https://doi.org/{doi}" if doi else "","open_access_url":None,"journal":journal,"concepts":[],"source":"crossref"})
        return papers
    except Exception as e:
        logger.warning(f"Crossref search failed: {e}")
        return []


async def _search_arxiv(query,discipline,year_from,year_to,limit,client):
    cat_map={"aerospace":"cat:cond-mat.mtrl-sci OR cat:physics.flu-dyn","materials":"cat:cond-mat.mtrl-sci","textile":"cat:cond-mat.soft","all":""}
    cat=cat_map.get(discipline,"")
    sq=f"all:{query}"
    if cat:sq=f"({sq}) AND ({cat})"
    try:
        resp=await client.get("https://export.arxiv.org/api/query",params={"search_query":sq,"start":0,"max_results":min(limit,15),"sortBy":"relevance","sortOrder":"descending"},timeout=15)
        if resp.status_code!=200:return []
        ns={"atom":"http://www.w3.org/2005/Atom","arxiv":"http://arxiv.org/schemas/atom"}
        root=ET.fromstring(resp.text)
        papers=[]
        for entry in root.findall("atom:entry",ns):
            arxiv_id=(entry.findtext("atom:id","",ns) or "").split("/abs/")[-1]
            title=(entry.findtext("atom:title","",ns) or "").replace("\n"," ").strip()
            abstract=(entry.findtext("atom:summary","",ns) or "").replace("\n"," ").strip()
            if not title:continue
            published=entry.findtext("atom:published","",ns) or ""
            year=int(published[:4]) if len(published)>=4 else None
            if year_from and year and year<year_from:continue
            if year_to and year and year>year_to:continue
            authors=[Author(name=a.findtext("atom:name","",ns) or "Unknown") for a in entry.findall("atom:author",ns)][:10]
            doi_tag=entry.findtext("arxiv:doi",None,ns)
            doi_url=f"https://doi.org/{doi_tag}" if doi_tag else ""
            papers.append({"paper_id":f"AR_{arxiv_id.replace('.','_')}","title":title,"authors":authors,"year":year,"abstract":abstract or None,"citation_count":0,"doi":doi_url,"url":doi_url or f"https://arxiv.org/abs/{arxiv_id}","open_access_url":f"https://arxiv.org/pdf/{arxiv_id}","journal":"arXiv preprint","concepts":[],"source":"arxiv"})
        return papers
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []


async def _search_pubmed(query,discipline,year_from,year_to,limit,client):
    if discipline=="aerospace":return []
    search_params={"db":"pubmed","term":f"{query} AND (composite OR textile OR fibre OR fiber)","retmax":min(limit,15),"retmode":"json","sort":"relevance","tool":"TriFieldAI","email":"contact@trifield.ai"}
    if year_from:search_params["mindate"]=str(year_from)
    if year_to:search_params["maxdate"]=str(year_to)
    try:
        resp=await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",params=search_params,timeout=12)
        if resp.status_code!=200:return []
        pmids=resp.json().get("esearchresult",{}).get("idlist",[])
        if not pmids:return []
        fetch_resp=await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",params={"db":"pubmed","id":",".join(pmids),"retmode":"xml","tool":"TriFieldAI","email":"contact@trifield.ai"},timeout=12)
        if fetch_resp.status_code!=200:return []
        root=ET.fromstring(fetch_resp.text)
        papers=[]
        for article in root.findall(".//PubmedArticle"):
            medline=article.find("MedlineCitation")
            if medline is None:continue
            art=medline.find("Article")
            if art is None:continue
            pmid=(medline.findtext("PMID") or "").strip()
            title=(art.findtext("ArticleTitle") or "").strip()
            if not title:continue
            abs_el=art.find("Abstract/AbstractText")
            abstract=abs_el.text.strip() if abs_el is not None and abs_el.text else None
            pub_date=art.find(".//PubDate")
            year=None
            if pub_date is not None:
                yr=pub_date.findtext("Year")
                year=int(yr) if yr and yr.isdigit() else None
            authors=[]
            for a in art.findall("AuthorList/Author")[:10]:
                last=a.findtext("LastName") or ""
                fore=a.findtext("ForeName") or ""
                name=f"{fore} {last}".strip()
                if name:authors.append(Author(name=name))
            journal=art.findtext("Journal/Title")
            doi=""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType")=="doi":doi=id_el.text or "";break
            papers.append({"paper_id":f"PM_{pmid}","title":title,"authors":authors,"year":year,"abstract":abstract,"citation_count":0,"doi":f"https://doi.org/{doi}" if doi else "","url":f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/","open_access_url":f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/" if pmid else None,"journal":journal,"concepts":[],"source":"pubmed"})
        return papers
    except Exception as e:
        logger.warning(f"PubMed search failed: {e}")
        return []


def _deduplicate(papers: list[dict]) -> list[dict]:
    """
    BUG FIX: original code iterated the input list at the end, allowing
    previously-merged duplicates to re-appear via their title-alias key.
    Now we iterate by_key.values() directly — the already-deduplicated set.
    """
    by_doi:   dict[str, dict] = {}
    by_title: dict[str, dict] = {}

    for p in papers:
        doi    = (p.get("doi") or "").strip()
        ntitle = _norm(p.get("title") or "")

        # Merge into existing entry if same DOI
        if doi and doi in by_doi:
            existing = by_doi[doi]
            if not existing.get("abstract") and p.get("abstract"):
                existing["abstract"] = p["abstract"]
            if not existing.get("open_access_url") and p.get("open_access_url"):
                existing["open_access_url"] = p["open_access_url"]
            if (p.get("citation_count") or 0) > (existing.get("citation_count") or 0):
                existing["citation_count"] = p["citation_count"]
            if p["source"] in ("crossref","pubmed") and p.get("journal"):
                existing["journal"] = p["journal"]
            # keep title alias pointing to merged entry
            if ntitle and ntitle not in by_title:
                by_title[ntitle] = existing
            continue

        # Merge into existing entry if same normalised title (different source)
        if ntitle and ntitle in by_title:
            existing = by_title[ntitle]
            if not existing.get("abstract") and p.get("abstract"):
                existing["abstract"] = p["abstract"]
            if not existing.get("open_access_url") and p.get("open_access_url"):
                existing["open_access_url"] = p["open_access_url"]
            if (p.get("citation_count") or 0) > (existing.get("citation_count") or 0):
                existing["citation_count"] = p["citation_count"]
            if p["source"] in ("crossref","pubmed") and p.get("journal"):
                existing["journal"] = p["journal"]
            if doi:
                by_doi[doi] = existing
            continue

        # New entry
        if doi:
            by_doi[doi] = p
        if ntitle:
            by_title[ntitle] = p

    # Collect unique papers — prefer DOI-keyed entries, fall back to title-keyed
    seen: set[int] = set()
    result: list[dict] = []
    for p in list(by_doi.values()) + [p for p in by_title.values() if id(p) not in {id(x) for x in by_doi.values()}]:
        if id(p) not in seen:
            seen.add(id(p))
            result.append(p)
    return result


# ── Main search ────────────────────────────────────────────────────────────

async def search_papers(
    query: str,
    discipline: str = "all",
    year_from: int | None = None,
    year_to:   int | None = None,
    limit:     int = 10,
) -> tuple[list[Paper], dict]:
    """
    Returns (papers, query_meta) where query_meta includes rewrite info.
    """
    key    = _cache_key(query, discipline, year_from, year_to, limit)
    cached = _get_cached(key)
    if cached is not None:
        return cached, {"source": "cache"}

    # LLM query rewriting (Groq — fast)
    rewrite = await rewrite_query(query, discipline)
    effective_discipline = rewrite.get("discipline", discipline)
    if discipline != "all":
        effective_discipline = discipline   # user override always wins

    # Use all 3 search query variants
    search_queries = rewrite.get("search_queries", [query])[:3]
    primary_query  = search_queries[0]

    async with httpx.AsyncClient(timeout=20) as client:
        # Run primary query across all sources
        raw_results = await asyncio.gather(
            _search_openalex(primary_query, effective_discipline, year_from, year_to, limit, client),
            _search_crossref(primary_query, effective_discipline, year_from, year_to, limit, client),
            _search_arxiv(   primary_query, effective_discipline, year_from, year_to, limit, client),
            _search_pubmed(  primary_query, effective_discipline, year_from, year_to, limit, client),
            return_exceptions=True
        )

        all_raw: list[dict] = []
        for r in raw_results:
            if isinstance(r, list): all_raw.extend(r)

        # Secondary query if not enough results
        if len(all_raw) < limit and len(search_queries) > 1:
            sec_results = await asyncio.gather(
                *[_search_openalex(sq, effective_discipline, year_from, year_to, limit//2, client)
                  for sq in search_queries[1:]],
                return_exceptions=True
            )
            for r in sec_results:
                if isinstance(r, list): all_raw.extend(r)

        # Clean
        clean = [p for p in all_raw if not _is_junk(p) and not _is_excluded(p.get("concepts") or [])]

        # Deduplicate
        merged = _deduplicate(clean)

        # Quality scoring
        query_terms = rewrite.get("primary_keywords", query.lower().split())
        entities    = rewrite.get("primary_keywords", [])
        ranked      = rank_papers(merged, query_terms, entities, effective_discipline)

        # Enrich top candidates
        candidates = ranked[:limit * 2]
        enriched = await asyncio.gather(
            *[_enrich(p, client) for p in candidates],
            return_exceptions=True
        )
        final = [r if isinstance(r, dict) else o for o, r in zip(candidates, enriched)]

        # Re-score after enrichment (abstracts now available)
        final_ranked = rank_papers(final, query_terms, entities, effective_discipline)
        top = final_ranked[:limit]

    papers = []
    for p in top:
        abstract = p.get("abstract")
        concepts = p.get("concepts") or []
        papers.append(Paper(
            paper_id        = p["paper_id"],
            title           = p.get("title") or "Untitled",
            authors         = p.get("authors") or [],
            year            = p.get("year"),
            abstract        = abstract,
            citation_count  = p.get("citation_count") or 0,
            url             = p.get("url") or "",
            open_access_url = p.get("open_access_url"),
            journal         = p.get("journal"),
            discipline_tag  = _tag_discipline(p.get("title",""), abstract, concepts),
        ))

    _set_cache(key, papers)
    return papers, rewrite
