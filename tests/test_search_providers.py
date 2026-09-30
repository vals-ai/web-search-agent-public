import logging
import unittest
from functools import partial
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer
from tavily import AsyncTavilyClient

from web_search_agent import tools as agent_tools
from web_search_agent.search_providers import ProviderWebSearch, format_result


class SearchProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests = []
        self.response = {}
        self.status = 200
        self.logger = logging.getLogger("test.search")

        async def search(request):
            self.requests.append(
                ({name.lower(): value for name, value in request.headers.items()}, await request.json())
            )
            response = {"search_id": "search_test", "session_id": "session_test", **self.response}
            return web.json_response(response, status=self.status)

        application = web.Application()
        application.router.add_post("/v1/search", search)
        application.router.add_post("/search", search)
        self.server = TestServer(application)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.base_url = str(self.server.make_url("/"))

    async def execute_tavily(self, **options):
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-secret"}):
            client = partial(AsyncTavilyClient, api_base_url=self.base_url)
            with patch.object(agent_tools, "AsyncTavilyClient", client):
                tool = agent_tools.TavilyWebSearch(**options)
                try:
                    return await tool.execute({"search_query": "annual revenue"}, {}, self.logger)
                finally:
                    await tool._client.close()

    async def test_each_tavily_depth_reaches_the_sdk_and_preserves_results(self):
        self.response = {
            "results": [{"title": "Filing", "url": "https://example.com/filing", "content": "Revenue: $42m"}]
        }

        for depth in ("basic", "advanced", "fast", "ultra-fast"):
            with self.subTest(depth=depth):
                result = await self.execute_tavily(search_depth=depth)

                self.assertIsNone(result.error)
                self.assertIn("https://example.com/filing", result.output)
                self.assertIn("Revenue: $42m", result.output)
                headers, body = self.requests[-1]
                self.assertEqual(headers["authorization"], "Bearer test-secret")
                self.assertEqual(
                    body, {"query": "annual revenue", "search_depth": depth, "max_results": 10, "chunks_per_source": 1}
                )

    async def test_invalid_tavily_depth_fails_before_any_request(self):
        with self.assertRaisesRegex(ValueError, "depth"):
            await self.execute_tavily(search_depth="not-a-depth")

        self.assertEqual(self.requests, [])


class FakeWebSearch(ProviderWebSearch):
    def __init__(self, search) -> None:
        super().__init__("fake")
        self._search = search

    async def search(self, query: str) -> str:
        return await self._search(query, self._api_key)


class ProviderWebSearchTests(unittest.IsolatedAsyncioTestCase):
    logger = logging.getLogger("test.search")

    async def execute(self, search, args=None):
        with patch.dict("os.environ", {"FAKE_API_KEY": "test-secret"}):
            tool = FakeWebSearch(search)
        return await tool.execute({"search_query": "annual revenue"} if args is None else args, {}, self.logger)

    async def test_results_reach_the_model_with_the_query_and_key(self):
        async def search(query, api_key):
            return format_result("Filing", "https://example.com/filing", f"{query} with {len(api_key)}-char key")

        result = await self.execute(search)

        self.assertIsNone(result.error)
        self.assertEqual(result.output, "### Filing\nhttps://example.com/filing\nannual revenue with 11-char key")

    async def test_provider_errors_are_tool_errors_that_do_not_expose_keys(self):
        async def search(query, api_key):
            raise RuntimeError(f"rejected {api_key}")

        with self.assertLogs(self.logger, level="WARNING") as logs:
            result = await self.execute(search)

        self.assertEqual(result.error, "fake search failed (RuntimeError).")
        self.assertNotIn("test-secret", result.output + str(logs.output))

    async def test_result_without_url_is_an_error(self):
        async def search(query, api_key):
            return format_result("Missing URL", " ", "text")

        with self.assertLogs(self.logger, level="WARNING"):
            result = await self.execute(search)

        self.assertIsNotNone(result.error)

    async def test_empty_query_is_rejected_without_searching(self):
        async def search(query, api_key):
            raise AssertionError("searched")

        result = await self.execute(search, {"search_query": "  "})

        self.assertEqual(result.error, "Invalid search query")

    def test_missing_credentials_fail_at_construction(self):
        with patch.dict("os.environ", {"FAKE_API_KEY": ""}), self.assertRaisesRegex(ValueError, "FAKE_API_KEY"):
            FakeWebSearch(None)


if __name__ == "__main__":
    unittest.main()
