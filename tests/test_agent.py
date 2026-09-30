import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from model_library.base import TextInput
from model_library.base.input import SystemInput

from web_search_agent.agent import SYSTEM_PROMPTS, Parameters, build_input, get_agent
from web_search_agent.dataset import load_tasks
from web_search_agent.tools import TavilyWebSearch

ROOT = Path(__file__).resolve().parents[1]


class AgentTests(unittest.TestCase):
    def test_agent_has_tavily_search_calculator_and_submission(self):
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}),
            patch("web_search_agent.agent.Agent") as constructor,
        ):
            get_agent(Parameters(model_name="openai/gpt-5.4-2026-03-05"), llm=Mock())

        tools = constructor.call_args.kwargs["tools"]
        self.assertIsInstance(tools[0], TavilyWebSearch)
        self.assertEqual([tool.name for tool in tools], ["web_search", "calculator", "submit_final_result"])

    def test_system_prompt_follows_the_domain(self):
        for domain in ("finance", "legal"):
            with self.subTest(domain=domain):
                system, question = build_input("What changed?", domain)
                assert isinstance(system, SystemInput) and isinstance(question, TextInput)
                self.assertEqual(system.text, SYSTEM_PROMPTS[domain])
                self.assertEqual(question.text, "Question:\nWhat changed?")
        self.assertNotEqual(SYSTEM_PROMPTS["finance"], SYSTEM_PROMPTS["legal"])


class DatasetTests(unittest.TestCase):
    def test_public_datasets_load_with_their_domain_and_unique_ids(self):
        for name, domain, count in (("finance", "finance", 27), ("legal", "legal", 5)):
            with self.subTest(dataset=name):
                dataset = load_tasks(ROOT / "data" / f"{name}.json")
                self.assertEqual(dataset.domain, domain)
                self.assertEqual(len(dataset.tests), count)
                self.assertEqual(len({task.id for task in dataset.tests}), count)
                self.assertTrue(all(task.question.strip() for task in dataset.tests))

    def test_unknown_domain_is_rejected(self):
        path = ROOT / "tests" / "_bad_dataset.json"
        path.write_text(json.dumps({"dataset_name": "x", "domain": "medical", "tests": []}))
        self.addCleanup(path.unlink)
        with self.assertRaises(ValueError):
            load_tasks(path)


if __name__ == "__main__":
    unittest.main()
