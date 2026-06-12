"""
Usage analytics tracker — in-memory with optional persistence.
Tracks: searches, latency, failed queries, top topics, PDF uploads.
"""
import time
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchEvent:
    query:        str
    discipline:   str
    result_count: int
    latency_ms:   float
    intent:       str
    success:      bool
    timestamp:    float = field(default_factory=time.time)


@dataclass
class PDFEvent:
    filename:    str
    session_id:  str
    chunk_count: int
    latency_ms:  float
    timestamp:   float = field(default_factory=time.time)


class AnalyticsTracker:
    def __init__(self):
        self.search_events: list[SearchEvent] = []
        self.pdf_events:    list[PDFEvent]    = []
        self._query_counts:    Counter = Counter()
        self._failed_queries:  Counter = Counter()
        self._discipline_counts: Counter = Counter()
        self._intent_counts:   Counter = Counter()

    # ── Record events ──────────────────────────────────────────────────────

    def record_search(
        self,
        query: str,
        discipline: str,
        result_count: int,
        latency_ms: float,
        intent: str = "general",
        success: bool = True,
    ):
        event = SearchEvent(
            query=query, discipline=discipline, result_count=result_count,
            latency_ms=latency_ms, intent=intent, success=success,
        )
        self.search_events.append(event)
        self._query_counts[query.lower()] += 1
        self._discipline_counts[discipline] += 1
        self._intent_counts[intent] += 1
        if not success:
            self._failed_queries[query.lower()] += 1

        # Keep last 10,000 events
        if len(self.search_events) > 10_000:
            self.search_events = self.search_events[-10_000:]

    def record_pdf(self, filename: str, session_id: str, chunk_count: int, latency_ms: float):
        self.pdf_events.append(PDFEvent(
            filename=filename, session_id=session_id,
            chunk_count=chunk_count, latency_ms=latency_ms,
        ))
        if len(self.pdf_events) > 5_000:
            self.pdf_events = self.pdf_events[-5_000:]

    # ── Stats ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        if not self.search_events:
            return {"message": "No search events recorded yet."}

        latencies  = [e.latency_ms for e in self.search_events]
        successful = [e for e in self.search_events if e.success]
        failed     = [e for e in self.search_events if not e.success]
        avg_results = (
            sum(e.result_count for e in successful) / len(successful)
            if successful else 0
        )

        return {
            "total_searches":      len(self.search_events),
            "successful_searches": len(successful),
            "failed_searches":     len(failed),
            "success_rate_pct":    round(len(successful) / len(self.search_events) * 100, 1),
            "avg_latency_ms":      round(sum(latencies) / len(latencies), 1),
            "p95_latency_ms":      round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
            "avg_results_per_search": round(avg_results, 1),
            "top_queries":         self._query_counts.most_common(10),
            "top_disciplines":     self._discipline_counts.most_common(5),
            "top_intents":         self._intent_counts.most_common(5),
            "top_failed_queries":  self._failed_queries.most_common(5),
            "total_pdf_uploads":   len(self.pdf_events),
        }


# ── Singleton ──────────────────────────────────────────────────────────────
_tracker: Optional[AnalyticsTracker] = None

def get_tracker() -> AnalyticsTracker:
    global _tracker
    if _tracker is None:
        _tracker = AnalyticsTracker()
    return _tracker
