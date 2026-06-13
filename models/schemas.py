from pydantic import BaseModel
from typing import Optional, Any


# ── Search ────────────────────────────────────────────────────────────────
class Author(BaseModel):
    name: str

class Paper(BaseModel):
    paper_id:        str
    title:           str
    authors:         list[Author]
    year:            Optional[int]
    abstract:        Optional[str]
    citation_count:  Optional[int]
    url:             Optional[str]
    open_access_url: Optional[str]
    journal:         Optional[str]
    discipline_tag:  Optional[str]

class SearchResponse(BaseModel):
    query:               str
    interpreted_query:   Optional[str] = None
    intent:              Optional[str] = None
    detected_discipline: Optional[str] = None
    rewrite_source:      Optional[str] = None
    total:               int
    discipline:          str
    papers:              list[Paper]


# ── PDF Chat ──────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question:   str

class ChatMessage(BaseModel):
    role:    str
    content: str

class CitationSource(BaseModel):
    ref:         int
    page:        int
    page_height: Optional[float] = None        # PDF page height in points
    page_width:  Optional[float] = None        # PDF page width in points
    bbox:        Optional[list[float]] = None  # first-line bbox [x0,y0,x1,y1] — fallback
    bboxes:      Optional[list[list[float]]] = None  # Phase 5: ALL line bboxes for multi-highlight
    snippet:     str

class ChatResponse(BaseModel):
    session_id: str
    answer:     str
    sources:    list[CitationSource]
    history:    list[ChatMessage]


# ── Property extraction ───────────────────────────────────────────────────
class PropertyRow(BaseModel):
    property_name: str
    value:         str
    unit:          Optional[str]
    test_standard: Optional[str]
    page_ref:      Optional[Any]

class PropertyExtractionResponse(BaseModel):
    session_id: str
    properties: list[PropertyRow]


# ── Citations ─────────────────────────────────────────────────────────────
class CitationRequest(BaseModel):
    paper_id: Optional[str]       = None
    title:    Optional[str]       = None
    authors:  Optional[list[str]] = None
    year:     Optional[int]       = None
    journal:  Optional[str]       = None
    volume:   Optional[str]       = None
    pages:    Optional[str]       = None
    doi:      Optional[str]       = None
    style:    str                 = "apa"

class CitationResponse(BaseModel):
    style:    str
    citation: str
    paper_id: Optional[str]
