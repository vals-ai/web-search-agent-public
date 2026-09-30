import asyncio
import logging
import os
from abc import abstractmethod
from typing import Any, override

from model_library.agent import Tool, ToolOutput

SEARCH_TIMEOUT_SECONDS = 60


def format_result(title: str | None, url: str, content: str) -> str:
    if not url.strip():
        raise ValueError("Search result is missing its citation URL")

    return f"### {title or '(no title)'}\n{url}\n{content}"


class ProviderWebSearch(Tool):
    name = "web_search"
    description = (
        "Search the public internet for up-to-date information. "
        "Provide a specific search query. Results contain a title, URL, and page excerpts."
    )
    parameters: dict[str, Any] = {"search_query": {"type": "string", "description": "The specific search query to run"}}
    required: list[str] = ["search_query"]

    def __init__(self, provider: str) -> None:
        variable_name = f"{provider.upper()}_API_KEY"
        self._api_key = os.environ.get(variable_name, "").strip()
        if not self._api_key:
            raise ValueError(f"{variable_name} is required")

        self._provider = provider

    @abstractmethod
    async def search(self, query: str) -> str: ...

    @override
    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        query: object = args.get("search_query")
        if not isinstance(query, str) or not query.strip():
            return ToolOutput(output="A non-empty search_query is required.", error="Invalid search query")

        try:
            async with asyncio.timeout(SEARCH_TIMEOUT_SECONDS):
                return ToolOutput(output=await self.search(query))
        except Exception as error:
            # Provider bodies and exception messages can contain credentials.
            message = f"{self._provider} search failed ({type(error).__name__})."
            logger.warning(message)
            return ToolOutput(output=message, error=message)
