"""Crawl4AI web extract plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider


def register(ctx) -> None:
    """Register the Crawl4AI provider with the plugin context."""
    ctx.register_web_search_provider(Crawl4AIWebSearchProvider())
