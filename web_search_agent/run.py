import argparse
import asyncio
import json
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv
from model_library.base import LLMConfig

from .agent import DEFAULT_MAX_TIME_SECONDS, Domain, Parameters, build_input, get_agent
from .dataset import Task, load_tasks
from .tools import TavilySearchDepth

# A time limit is reported through the stop reason. Anything else that is not the
# submit tool is an error, so an incomplete run is never read as a finished one.
_STOP_REASON_STATUS = {"done_tool": "success", "max_time": "max_time"}


async def run_task(
    task: Task, domain: Domain, parameters: Parameters, output_dir: Path, log_dir: Path
) -> dict[str, object]:
    agent = get_agent(parameters, log_dir=log_dir)
    result = await agent.run(build_input(task.question, domain), question_id=task.id, atif_export=True)
    record: dict[str, object] = {
        "id": task.id,
        "domain": domain,
        "model": parameters.model_name,
        "tavily_search_depth": parameters.tavily_search_depth,
        "status": _STOP_REASON_STATUS.get(result.stop_reason.value, "error"),
        "question": task.question,
        "answer": result.final_answer,
        "result": result.model_dump(mode="json"),
    }
    path = output_dir / f"{task.id}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{task.id}: {record['status']} -> {path}")
    return record


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the web search agent on a finance or legal dataset")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSON, e.g. data/finance.json")
    parser.add_argument("--model", required=True, help="model-library model name, e.g. openai/gpt-5.4-2026-03-05")
    parser.add_argument("--tavily-search-depth", choices=get_args(TavilySearchDepth.__value__), default="advanced")
    parser.add_argument("--task-ids", nargs="+", help="Only run these task ids")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of tasks to run at once")
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME_SECONDS, help="Seconds per task")
    parser.add_argument("--max-tokens", type=int, help="Completion token limit (default: model-library's)")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (default: model-library's)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()

    load_dotenv(override=True, dotenv_path=Path(".env"))

    dataset = load_tasks(args.dataset)
    tasks = dataset.tests
    if args.task_ids:
        unknown = set(args.task_ids) - {task.id for task in tasks}
        if unknown:
            parser.error(f"unknown task ids: {', '.join(sorted(unknown))}")
        tasks = [task for task in tasks if task.id in args.task_ids]

    llm_kwargs = {
        name: value
        for name, value in (("max_tokens", args.max_tokens), ("temperature", args.temperature))
        if value is not None
    }
    parameters = Parameters(
        model_name=args.model,
        llm_config=LLMConfig(**llm_kwargs),
        max_time_seconds=args.max_time,
        tavily_search_depth=args.tavily_search_depth,
    )

    output_dir: Path = args.output_dir / dataset.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(task: Task) -> dict[str, object]:
        async with semaphore:
            return await run_task(task, dataset.domain, parameters, output_dir, args.log_dir)

    records = await asyncio.gather(*(bounded(task) for task in tasks))
    summary = output_dir / "answers.json"
    summary.write_text(
        json.dumps([{k: r[k] for k in ("id", "status", "answer")} for r in records], indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} answers to {summary}")


def main_sync() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
