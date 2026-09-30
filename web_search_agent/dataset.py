import json
from pathlib import Path

from pydantic import BaseModel

from .agent import Domain


class Task(BaseModel):
    id: str
    question: str


class Dataset(BaseModel):
    dataset_name: str
    domain: Domain
    tests: list[Task]


def load_tasks(path: Path) -> Dataset:
    return Dataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
