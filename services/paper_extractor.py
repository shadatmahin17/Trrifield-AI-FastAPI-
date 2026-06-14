"""
LLM-powered extraction of all 18 library column types from PDF text.
Each column is extracted on-demand and cached in PostgreSQL.
"""
import json, logging
from core.llm import llm_call

logger = logging.getLogger(__name__)

# All supported column types with their extraction prompts
COLUMN_DEFINITIONS = {
    "tldr": {
        "label":  "TL;DR",
        "prompt": "Write a 2-3 sentence plain-language summary of this paper. Be concrete and specific about what was done and what was found.",
    },
    "summarized_abstract": {
        "label":  "Summarized Abstract",
        "prompt": "Summarize the abstract in 2-3 sentences in plain language, preserving key numbers and findings.",
    },
    "results": {
        "label":  "Results",
        "prompt": "List the main quantitative and qualitative results as bullet points. Include specific numbers, percentages, and comparisons where mentioned.",
    },
    "summarized_introduction": {
        "label":  "Summarized Introduction",
        "prompt": "Summarize the introduction in 3-4 sentences. What problem does this paper address and why does it matter?",
    },
    "methods_used": {
        "label":  "Methods Used",
        "prompt": "List the key methods, techniques, and tools used in this study as bullet points (e.g. fabrication methods, characterization techniques, statistical tests).",
    },
    "literature_survey": {
        "label":  "Literature Survey",
        "prompt": "Summarize the key prior works referenced in this paper and what gap they left that this paper fills. 3-4 sentences.",
    },
    "limitations": {
        "label":  "Limitations",
        "prompt": "List the limitations explicitly mentioned or implied in this paper as bullet points.",
    },
    "contributions": {
        "label":  "Contributions",
        "prompt": "List the novel contributions of this paper as bullet points. What is new compared to prior work?",
    },
    "practical_implications": {
        "label":  "Practical Implications",
        "prompt": "What are the practical real-world applications and implications of this research? 2-3 sentences.",
    },
    "objectives": {
        "label":  "Objectives",
        "prompt": "List the research objectives or aims of this study as bullet points.",
    },
    "findings": {
        "label":  "Findings",
        "prompt": "List the key findings of this paper as bullet points. Be specific with numbers where available.",
    },
    "research_gap": {
        "label":  "Research Gap",
        "prompt": "What research gap does this paper identify and address? 2-3 sentences.",
    },
    "future_research": {
        "label":  "Future Research",
        "prompt": "List the future research directions suggested by the authors as bullet points.",
    },
    "dependent_variables": {
        "label":  "Dependent Variables",
        "prompt": "List the dependent variables (outcomes measured) in this study as bullet points.",
    },
    "independent_variables": {
        "label":  "Independent Variables",
        "prompt": "List the independent variables (factors manipulated or studied) as bullet points.",
    },
    "dataset": {
        "label":  "Dataset",
        "prompt": "Describe the dataset, materials, or samples used in this study. Include sample sizes, sources, and key characteristics.",
    },
    "population_sample": {
        "label":  "Population / Sample",
        "prompt": "Describe the study population or sample used. Include size, selection criteria, and characteristics.",
    },
    "problem_statement": {
        "label":  "Problem Statement",
        "prompt": "What is the core problem this paper is trying to solve? 1-2 sentences.",
    },
    "challenges": {
        "label":  "Challenges",
        "prompt": "List the key challenges or difficulties encountered or identified in this study as bullet points.",
    },
    "applications": {
        "label":  "Applications",
        "prompt": "List the potential application areas for this research as bullet points.",
    },
}

SYSTEM_PROMPT = """You are a research paper analysis assistant specialising in materials science, aerospace engineering, and textile engineering.
Analyse the provided paper text and answer the extraction task precisely.
Be concise, specific, and technical. Use bullet points where instructed.
If the information is not available in the provided text, respond with exactly: "Not available in this paper."
Never fabricate information."""


async def extract_column(paper_text: str, column_key: str) -> str:
    """
    Extract a single column value from paper text using LLM.
    Returns the extracted text string.
    """
    if column_key not in COLUMN_DEFINITIONS:
        raise ValueError(f"Unknown column key: {column_key}")

    col    = COLUMN_DEFINITIONS[column_key]
    prompt = f"{col['prompt']}\n\nPAPER TEXT:\n{paper_text[:6000]}"

    try:
        result = await llm_call(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            task="research_summary",
        )
        return result.strip()
    except Exception as e:
        logger.error(f"Column extraction failed for {column_key}: {e}")
        return "Extraction failed. Please try again."


async def extract_metadata_from_text(paper_text: str) -> dict:
    """
    Auto-extract title, authors, abstract, DOI, journal, year from paper text.
    Used on upload to populate library metadata.
    """
    prompt = f"""Extract metadata from this research paper. Return ONLY valid JSON, no markdown:
{{
  "title": "full paper title or null",
  "authors": ["Author 1", "Author 2"],
  "abstract": "full abstract text or null",
  "doi": "DOI string without https://doi.org/ prefix or null",
  "journal": "journal name or null",
  "year": 2024
}}

PAPER TEXT (first 3000 chars):
{paper_text[:3000]}"""

    try:
        raw = await llm_call(
            system="You are a metadata extraction specialist. Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            prefer_json=True,
            task="property_extract",
        )
        raw = raw.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
        return {"title": None, "authors": [], "abstract": None,
                "doi": None, "journal": None, "year": None}
