import json
import logging
import math
import os
import re
from typing import Any, Literal, get_args

import backoff
from model_library.agent import Tool, ToolOutput
from simpleeval import SimpleEval
from tavily import AsyncTavilyClient, UsageLimitExceededError

_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

type TavilySearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


class TavilyWebSearch(Tool):
    name = "web_search"
    description = (
        "Search the public internet for information. Each result will contain "
        "a url, a title, and one excerpt taken directly from the page."
    )
    parameters: dict[str, Any] = {
        "search_query": {
            "type": "string",
            "description": "The query to search for",
        },
        "start_date": {
            "type": "string",
            "description": "(optional) The start date for the search range in the format YYYY-MM-DD",
        },
        "end_date": {
            "type": "string",
            "description": "(optional) The end date for the search range in the format YYYY-MM-DD",
        },
        "number_of_results": {
            "type": "integer",
            "description": "(optional) The number of search results to return.",
            "maximum": 20,
            "minimum": 1,
            "default": 10,
        },
    }
    required: list[str] = ["search_query"]

    def __init__(self, tavily_api_key: str | None = None, search_depth: TavilySearchDepth = "advanced") -> None:
        if search_depth not in get_args(TavilySearchDepth.__value__):
            raise ValueError(f"Unsupported Tavily search depth: {search_depth}")

        self._client = AsyncTavilyClient(api_key=tavily_api_key or os.environ["TAVILY_API_KEY"])
        self._search_depth: TavilySearchDepth = search_depth

    @backoff.on_exception(
        backoff.expo,
        UsageLimitExceededError,
        max_tries=4,
        max_time=60,
        max_value=15,
        base=2,
        factor=3,
        jitter=backoff.full_jitter,
    )
    async def _search(
        self,
        search_query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        number_of_results: int = 10,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if end_date is not None:
            if not _DATE_REGEX.match(end_date):
                raise ValueError(f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD.")
            kwargs["end_date"] = end_date
        if start_date is not None:
            if not _DATE_REGEX.match(start_date):
                raise ValueError(f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD.")
            if end_date is not None and start_date > end_date:
                raise ValueError(
                    f"Parameter start_date '{start_date}' was set to a date that is later than end_date '{end_date}'"
                )
            kwargs["start_date"] = start_date

        response = await self._client.search(
            search_depth=self._search_depth,
            max_results=number_of_results,
            chunks_per_source=1,
            query=search_query,
            **kwargs,
        )
        return response.get("results", [])

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        try:
            results = await self._search(**args)
            return ToolOutput(output=json.dumps(results, default=str))
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"web_search (tavily) failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class SubmitFinalResult(Tool):
    name = "submit_final_result"
    description = (
        "Submits the final answer to the user. You should include your final answer, as well as any necessary "
        "reasoning, justification, calculations, and explanation. Finally, you should provide any sources used to answer the question. "
        "You MUST use this tool to submit your final result. The user will not see your response if you do not use this tool to submit. "
        "You will not be able to continue working after this tool is called; the conversation will be ended."
    )
    parameters: dict[str, Any] = {
        "final_result": {
            "type": "string",
            "description": "The final result to submit to the user",
        }
    }
    required: list[str] = ["final_result"]

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        try:
            final_result = args["final_result"]
            if not final_result:
                raise ValueError("Final result must not be empty")
            return ToolOutput(output=final_result, done=True)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Submission failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg, done=False)


class Calculator(Tool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression and return the result. "
        "Use for all arithmetic instead of computing by hand. "
        "Supports: +, -, *, /, ** (exponentiation), % (modulo), parentheses. "
        "Functions: abs(), min(), max(), sqrt(), log(), log10(). "
        "Example: '(5000 - 3200) * 0.21'"
    )
    parameters: dict[str, Any] = {
        "expression": {
            "type": "string",
            "description": "The mathematical expression to evaluate",
        }
    }
    required: list[str] = ["expression"]

    def __init__(self) -> None:
        self._evaluator = SimpleEval(
            functions={
                "abs": abs,
                "min": min,
                "max": max,
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
            }
        )

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        expression = args.get("expression", "")
        if not expression:
            return ToolOutput(output="Error: expression must not be empty", error="empty expression")
        try:
            result = self._evaluator.eval(expression)
            return ToolOutput(output=str(result))
        except ZeroDivisionError:
            msg = f"Error: division by zero in '{expression}'"
            return ToolOutput(output=msg, error=msg)
        except Exception as e:
            msg = f"Error: invalid expression '{expression}': {e}"
            logger.warning(msg)
            return ToolOutput(output=msg, error=msg)
