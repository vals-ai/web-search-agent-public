from pathlib import Path
from typing import Literal

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    TimeLimit,
    Tool,
    TurnResult,
    default_before_query,
    truncate_oldest,
)
from model_library.base import LLM, LLMConfig, RawResponse, TextInput
from model_library.base.input import InputItem, SystemInput
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel

from .prompt import FINANCE_SYSTEM_PROMPT, LEGAL_SYSTEM_PROMPT, QUESTION_PROMPT
from .tools import Calculator, SubmitFinalResult, TavilySearchDepth, TavilyWebSearch

type Domain = Literal["finance", "legal"]

DEFAULT_MAX_TIME_SECONDS = 7200

SYSTEM_PROMPTS: dict[Domain, str] = {
    "finance": FINANCE_SYSTEM_PROMPT,
    "legal": LEGAL_SYSTEM_PROMPT,
}


class Parameters(BaseModel):
    model_name: str
    llm_config: LLMConfig = LLMConfig()
    max_time_seconds: int = DEFAULT_MAX_TIME_SECONDS
    tavily_search_depth: TavilySearchDepth = "advanced"


def build_input(question: str, domain: Domain) -> list[InputItem]:
    return [
        SystemInput(text=SYSTEM_PROMPTS[domain]),
        TextInput(text=QUESTION_PROMPT.format(question=question)),
    ]


def build_search_tool(parameters: Parameters) -> Tool:
    return TavilyWebSearch(search_depth=parameters.tavily_search_depth)


def get_agent(parameters: Parameters, llm: LLM | None = None, log_dir: Path | None = None) -> Agent:
    if llm is None:
        llm = get_registry_model(parameters.model_name, parameters.llm_config)

    tools: list[Tool] = [build_search_tool(parameters), Calculator(), SubmitFinalResult()]

    def _before_query(history: list[InputItem], last_error: Exception | None) -> list[InputItem]:
        if isinstance(last_error, MaxContextWindowExceededError):
            return truncate_oldest(history)

        if history and isinstance(history[-1], RawResponse):
            # Paused server tools must resume in the same assistant turn.
            if getattr(history[-1].response, "stop_reason", None) == "pause_turn":
                return default_before_query(history, last_error)

            history.append(
                TextInput(
                    text=(
                        "Your last response produced no tool call. "
                        "Call `submit_final_result` if you have a final result, "
                        "otherwise continue with the next tool call."
                    )
                )
            )

        return default_before_query(history, last_error)

    def _should_stop(turn_result: TurnResult) -> bool:
        return False

    return Agent(
        llm=llm,
        tools=tools,
        name="web-search",
        log_dir=log_dir or Path("logs"),
        config=AgentConfig(
            turn_limit=None,
            time_limit=TimeLimit(max_seconds=parameters.max_time_seconds),
        ),
        hooks=AgentHooks(before_query=_before_query, should_stop=_should_stop),
    )
