"""Async arXiv client: keyword extraction, Atom API search, optional PDF text extraction.

arXiv API docs: https://info.arxiv.org/help/api/user-manual.html
Etiquette: identify the client, keep requests small, no parallel hammering.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("teleport.arxiv")

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
USER_AGENT = "LatentTeleportation/1.0 (hackathon research prototype)"

# Target knowledge domain -> arXiv categories. Restricting the search to the *target* field's
# categories is what makes the result cross-domain rather than a paper about the source problem.
DOMAIN_CATEGORIES: Dict[str, List[str]] = {
    "natural": ["q-bio.PE", "q-bio.MN", "q-bio.CB", "q-bio.QM"],
    "physics": ["physics.flu-dyn", "cond-mat.stat-mech", "nlin.AO", "nlin.CD", "physics.comp-ph"],
    "social": ["physics.soc-ph", "q-bio.PE", "cs.SI", "econ.TH"],
    "math": ["math.OC", "math.PR", "math.DS", "math.CO"],
    "cognitive": ["q-bio.NC", "cs.NE"],
}

# Source-problem words -> structural mechanism vocabulary used in other sciences.
# Lets "autoscaler overshoots" search physics/biology for "oscillation feedback delay".
MECHANISM_VOCAB: List[Tuple[str, List[str]]] = [
    (r"congest|traffic|jam|bottleneck|throughput|backpressure", ["congestion", "flow"]),
    (r"queue|backlog|wait|overflow|triage", ["queueing", "load"]),
    (r"oscillat|overshoot|flap|thrash|unstable|instabil", ["oscillation", "feedback"]),
    (r"lock|mutex|contention|deadlock|starv", ["contention", "allosteric"]),
    (r"collid|collision|crash into|intersection|aisle", ["collision avoidance", "collective motion"]),
    (r"cascade|amplif|bullwhip|ripple|propagat", ["cascade", "amplification"]),
    (r"spread|viral|misinformation|rumou?r|contagi", ["contagion", "epidemic"]),
    (r"degrad|wear|aging|ageing|decay|fatigue", ["degradation", "aging"]),
    (r"bias|echo chamber|filter bubble|polariz", ["polarization", "opinion dynamics"]),
    (r"route|routing|path|network", ["transport network", "routing"]),
    (r"schedul|allocat|resource|capacity", ["resource allocation"]),
    (r"latency|delay|lag", ["delay"]),
    (r"synchron|consensus|coordinat", ["synchronization", "consensus"]),
    (r"liquidity|order book|slippage|volatil", ["liquidity", "volatility"]),
]

STOPWORDS = set(
    """a an and are as at be been but by can could do does for from has have how i in into is it its
    keep keeps more most my not of on or our so some such than that the their them then there these
    they this to too up us very was we were what when where which while who why will with would you
    your over under about across find analogue analog bottleneck problem issue system systems using use
    get gets got make makes made really just also like""".split()
)


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    url: str
    pdf_url: Optional[str]
    categories: List[str] = field(default_factory=list)
    pdf_text: str = ""


def extract_keywords(text: str, limit: int = 4) -> Tuple[List[str], List[str]]:
    """Returns (mechanism_terms, literal_terms) for building arXiv queries."""
    lowered = text.lower()
    mechanism: List[str] = []
    for pattern, terms in MECHANISM_VOCAB:
        if re.search(pattern, lowered):
            for t in terms:
                if t not in mechanism:
                    mechanism.append(t)
    words = re.findall(r"[a-z][a-z\-]{3,}", lowered)
    literal: List[str] = []
    for w in words:
        if w in STOPWORDS or w in literal:
            continue
        literal.append(w)
    return mechanism[:limit], literal[:limit]


def _quote(term: str) -> str:
    return f'all:"{term}"' if " " in term else f"all:{term}"


def build_queries(text: str, domain: str) -> List[str]:
    """Ordered from most to least specific; caller stops once it has enough papers."""
    mechanism, literal = extract_keywords(text)
    cats = DOMAIN_CATEGORIES.get(domain) or [c for cs in DOMAIN_CATEGORIES.values() for c in cs]
    cat_clause = "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"
    queries: List[str] = []
    if mechanism:
        queries.append("(" + " OR ".join(_quote(t) for t in mechanism) + f") AND {cat_clause}")
    if literal:
        queries.append("(" + " OR ".join(_quote(t) for t in literal[:3]) + f") AND {cat_clause}")
    if mechanism or literal:
        queries.append(" OR ".join(_quote(t) for t in (mechanism + literal)[:4]))
    return queries


def parse_atom(xml_text: str) -> List[ArxivPaper]:
    root = ET.fromstring(xml_text)
    papers: List[ArxivPaper] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
        if "/abs/" not in raw_id:
            continue  # arXiv returns an error pseudo-entry for malformed queries
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        pdf_url = None
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=" ".join((entry.findtext(f"{ATOM}title") or "").split()),
                abstract=" ".join((entry.findtext(f"{ATOM}summary") or "").split()),
                authors=[
                    (a.findtext(f"{ATOM}name") or "").strip() for a in entry.findall(f"{ATOM}author")
                ],
                published=(entry.findtext(f"{ATOM}published") or "")[:10],
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url,
                categories=[c.get("term", "") for c in entry.findall(f"{ATOM}category")],
            )
        )
    return papers


class ArxivClient:
    def __init__(self, timeout: float = 8.0, cache_ttl: float = 3600.0) -> None:
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, List[ArxivPaper]]] = {}
        self._lock = asyncio.Lock()  # serialize requests, per arXiv API etiquette

    async def search(self, query: str, max_results: int = 8) -> List[ArxivPaper]:
        key = f"{query}|{max_results}"
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_ttl:
            return hit[1]
        params = {
            "search_query": query,
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
        }
        async with self._lock:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
            ) as client:
                resp = await client.get(ARXIV_API, params=params)
                resp.raise_for_status()
        papers = parse_atom(resp.text)
        self._cache[key] = (time.monotonic(), papers)
        return papers

    async def fetch_pdf_text(self, paper: ArxivPaper, max_pages: int = 3, max_bytes: int = 15_000_000) -> str:
        """Downloads the paper PDF and extracts the first pages with pypdf. Empty string on failure."""
        if not paper.pdf_url:
            return ""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
            ) as client:
                resp = await client.get(paper.pdf_url)
                resp.raise_for_status()
                data = resp.content[:max_bytes]
            return await asyncio.to_thread(_pdf_to_text, data, max_pages)
        except Exception as exc:  # network, parse, or non-PDF response
            logger.info("PDF fetch failed for %s: %s", paper.arxiv_id, exc)
            return ""


def _pdf_to_text(data: bytes, max_pages: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: List[str] = []
    for page in reader.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    text = unicodedata.normalize("NFKC", " ".join(parts))
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)  # re-join words hyphenated across line breaks
    return " ".join(text.split())
