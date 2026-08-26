"""Crawl4AI extract-only provider — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.

Extract-only — Crawl4AI uses a headless browser to render pages and convert
them to clean Markdown. For search, pair with SearXNG or another search
backend via ``web.search_backend`` and set ``web.extract_backend: crawl4ai``.

Config keys this provider responds to::

    web:
      extract_backend: "crawl4ai"     # explicit per-capability
      backend: "crawl4ai"             # shared fallback (if search handled by other backend)

Env var::

    CRAWL4AI_URL=http://localhost:11235   # default if unset
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Default URL matches the documented Docker setup
_DEFAULT_CRAWL4AI_URL = "http://localhost:11235"


def _get_base_url() -> str:
    """Return the Crawl4AI API base URL, stripping trailing slashes."""
    url = os.getenv("CRAWL4AI_URL", _DEFAULT_CRAWL4AI_URL).strip().rstrip("/")
    return url


def _normalize_crawl4ai_results(
    raw_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Map Crawl4AI /crawl response to the standard extract document shape.

    Crawl4AI returns::

        {
            "success": true,
            "results": [
                {
                    "url": "...",
                    "html": "...",
                    "markdown": {
                        "raw_markdown": "...",
                        "markdown_with_citations": "...",
                        "fit_markdown": "...",
                    },
                    "metadata": {...},
                    "screenshot": null,
                },
                ...
            ]
        }

    We extract ``raw_markdown`` as ``content`` / ``raw_content`` and surface
    errors for any URL that didn't produce usable Markdown.
    """
    documents: List[Dict[str, Any]] = []
    for result in raw_results:
        url = result.get("url", "")
        error = result.get("error", "")

        if error:
            documents.append({
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": error,
                "metadata": {"sourceURL": url},
            })
            continue

        md = result.get("markdown", {})
        raw_md = ""
        if isinstance(md, dict):
            raw_md = md.get("raw_markdown", "") or ""
        elif isinstance(md, str):
            raw_md = md

        title = ""
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            title = (
                metadata.get("title", "")
                or metadata.get("og:title", "")
                or metadata.get("sourceURL", "")
                or url
            )

        documents.append({
            "url": url,
            "title": title,
            "content": raw_md,
            "raw_content": raw_md,
            "metadata": {
                "sourceURL": url,
                "title": title,
            },
        })
    return documents


class Crawl4AIWebSearchProvider(WebSearchProvider):
    """Crawl4AI self-hosted web extraction provider."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI"

    def is_available(self) -> bool:
        """Return True when ``CRAWL4AI_URL`` is set (or default works)."""
        url = _get_base_url()
        # Always consider available if CRAWL4AI_URL is explicitly set,
        # or if we're using the default (Docker container should be running).
        # The extract() call will surface connection errors if the server
        # isn't actually reachable.
        return bool(url)

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from URLs via Crawl4AI's headless browser.

        Sends a POST to ``<CRAWL4AI_URL>/crawl`` with the Crawl4AI API
        payload. Each URL is rendered with Chromium and converted to
        clean Markdown.

        Sync — uses ``httpx.post(...)``. Returns the standard list-of-results
        shape; per-URL failures become items with ``error``.
        """
        import httpx

        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]
        except Exception:
            pass

        base_url = _get_base_url()
        logger.info("Crawl4AI extract: %d URL(s) → %s", len(urls), base_url)

        payload = {
            "urls": urls,
            "browser_config": {
                "type": "BrowserConfig",
                "params": {"headless": True},
            },
            "crawler_config": {
                "type": "CrawlerRunConfig",
                "params": {"cache_mode": "bypass"},
            },
        }

        try:
            resp = httpx.post(
                f"{base_url}/crawl",
                json=payload,
                timeout=120,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError:
            logger.warning("Crawl4AI connection refused at %s", base_url)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": (
                        f"Crawl4AI server unreachable at {base_url}. "
                        "Ensure the Docker container is running: "
                        "docker run -d -p 11235:11235 --name crawl4ai "
                        "--shm-size=1g unclecode/crawl4ai:latest"
                    ),
                }
                for u in urls
            ]
        except httpx.HTTPStatusError as exc:
            logger.warning("Crawl4AI HTTP error: %s", exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": f"Crawl4AI returned HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                }
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Crawl4AI extract error: %s", exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": f"Crawl4AI extract failed: {exc}",
                }
                for u in urls
            ]

        if not data.get("success", False):
            error = data.get("error", "Unknown Crawl4AI error")
            return [
                {"url": u, "title": "", "content": "", "error": error}
                for u in urls
            ]

        results = data.get("results", [])
        return _normalize_crawl4ai_results(results)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Crawl4AI",
            "badge": "self-hosted",
            "tag": "Self-hosted headless browser extraction. Pair with SearXNG for search.",
            "env_vars": [
                {
                    "key": "CRAWL4AI_URL",
                    "prompt": "Crawl4AI API URL",
                    "default": "http://localhost:11235",
                    "url": "https://github.com/unclecode/crawl4ai",
                },
            ],
        }
