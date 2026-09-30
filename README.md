# Web Search Agent

The agent behind the [Vals AI Web Search Index](https://www.vals.ai/benchmarks). It is one small research agent, and the only thing that changes between runs is the web search tool. This repo runs it on the public questions from [Finance Agent Benchmark v2](https://github.com/vals-ai/finance-agent-v2) (27 questions) and [Legal Research Bench](https://github.com/vals-ai/legal-research-bench) (5 questions).

## The agent

The agent is built on [`model-library`](https://github.com/vals-ai/model-library)'s agent loop and has three tools:

- `web_search`: Tavily web search.
- `calculator`: evaluates arithmetic expressions.
- `submit_final_result`: ends the run with the final answer.

It has no document fetching, EDGAR, CourtListener, or history compaction. When the context window overflows, the oldest turns are dropped. Each task has a 2-hour limit and no turn limit. The system prompt depends on the dataset's domain: finance questions get the finance prompt and legal questions get the legal prompt (`web_search_agent/prompt.py`).

## Web search

`web_search` uses [Tavily](https://tavily.com) (`TAVILY_API_KEY`). `--tavily-search-depth` picks `basic`, `advanced` (default), `fast`, or `ultra-fast`.

To use a different search provider, subclass `ProviderWebSearch` in `web_search_agent/search_providers.py` and implement `search(query) -> str`. The base class reads `<PROVIDER>_API_KEY`, applies a 60-second timeout, and reports failures as tool errors without exposing the key. Then return your tool from `build_search_tool` in `web_search_agent/agent.py`.

## Running

```bash
make install
cp .env.example .env   # add your model and search keys

uv run web-search-agent --dataset data/finance.json --model openai/gpt-5.4-2026-03-05
uv run web-search-agent --dataset data/legal.json --model openai/gpt-5.4-2026-03-05 --task-ids P-001
```

Answers go to `results/<dataset>/<task id>.json`, with a summary in `results/<dataset>/answers.json`. Agent logs go to `logs/`. Use `--concurrency` to run several tasks at once.

## Datasets

`data/finance.json` and `data/legal.json` contain the public questions from the two source repos:

```json
{"dataset_name": "legal", "domain": "legal", "tests": [{"id": "P-001", "question": "..."}]}
```

Any file in this format can be run. `domain` selects the system prompt.

## Grading

This repo only generates answers. The finance rubrics are published in the Finance Agent Benchmark v2 repo. Grading follows each source benchmark: a panel of judge models scores finance answers against rubrics, and a single judge scores legal answers against rubrics and checks the cited sources.

## Development

```bash
make style
make typecheck
make test
```
