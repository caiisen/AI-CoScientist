"""OpenAlex literature retrieval helpers."""

import threading
import time
from typing import Optional

import requests
from loguru import logger


class OpenAlexClient:
    """Synchronous client for OpenAlex semantic work search."""

    BASE_URL = "https://api.openalex.org/works"
    SEMANTIC_QUERY_MAX = 2000
    PER_PAGE_MAX = 50

    def __init__(
        self,
        mailto: Optional[str] = None,
        timeout: tuple[float, float] = (5.0, 15.0),
    ) -> None:
        self.mailto = mailto
        self.timeout = timeout

    def semantic_search(
        self, query: str, per_page: int = 10
    ) -> list[dict]:
        """Return simplified OpenAlex semantic search results."""
        query = (query or "").strip()[: self.SEMANTIC_QUERY_MAX]
        per_page = min(max(per_page, 1), self.PER_PAGE_MAX)
        if not query:
            return []

        params: dict[str, str | int] = {
            "search.semantic": query,
            "per-page": per_page,
        }
        if self.mailto:
            params["mailto"] = self.mailto

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(f"OpenAlex semantic_search failed: {exc}")
            return []

        papers = []
        for work in data.get("results", []):
            papers.append(
                {
                    "title": work.get("title") or "",
                    "abstract": self._reconstruct_abstract(
                        work.get("abstract_inverted_index")
                    ),
                    "doi": work.get("doi") or "",
                }
            )
        return papers

    @staticmethod
    def _reconstruct_abstract(
        inverted_index: Optional[dict],
    ) -> str:
        """Convert OpenAlex abstract_inverted_index to readable text."""
        if not inverted_index:
            return ""

        position_to_word: dict[int, str] = {}
        for word, positions in inverted_index.items():
            for position in positions:
                position_to_word[position] = word

        return " ".join(
            position_to_word[position]
            for position in sorted(position_to_word)
        )


class RateLimiter:
    """Thread-safe limiter with a minimum interval between acquisitions."""

    def __init__(self, min_interval: float = 1.0) -> None:
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


class LiteratureCache:
    """Small in-memory cache keyed by normalized query string."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[list[dict]]:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: list[dict]) -> None:
        with self._lock:
            self._store[key] = value


class LiteratureService:
    """Application-level interface for retrieving literature context."""

    def __init__(
        self,
        top_n: int = 10,
        mailto: Optional[str] = None,
    ) -> None:
        self.top_n = min(max(top_n, 1), OpenAlexClient.PER_PAGE_MAX)
        self.client = OpenAlexClient(mailto=mailto)
        self.limiter = RateLimiter(min_interval=1.0)
        self.cache = LiteratureCache()

    def fetch_for_goal(self, research_goal: str) -> list[dict]:
        return self._fetch(research_goal)

    def fetch_for_hypothesis(self, hypothesis: object) -> list[dict]:
        query = (getattr(hypothesis, "title", "") or "").strip()
        if not query:
            query = (getattr(hypothesis, "text", "") or "")[:200]
        return self._fetch(query)

    def _fetch(self, query: str) -> list[dict]:
        query = (query or "").strip()[: OpenAlexClient.SEMANTIC_QUERY_MAX]
        if not query:
            return []

        cached = self.cache.get(query)
        if cached is not None:
            return cached

        self.limiter.acquire()
        papers = self.client.semantic_search(query, per_page=self.top_n)
        self.cache.set(query, papers)
        return papers
