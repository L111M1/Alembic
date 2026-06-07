import json
import os
import tempfile
import pytest
import yaml

from alembic.api.base import BaseAPIClient


class FakeAPI(BaseAPIClient):
    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        return json.dumps({
            "instruction": "test instruction here",
            "output": "test output data here for testing",
        })


class FakeScoreAPI(BaseAPIClient):
    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        return json.dumps({
            "correctness": 9,
            "helpfulness": 8,
            "completeness": 7,
            "clarity": 8,
        })


@pytest.fixture
def fake_api():
    return FakeAPI()


@pytest.fixture
def fake_score_api():
    return FakeScoreAPI()


@pytest.fixture
def temp_jsonl():
    files = []

    def _make(lines: list[str]) -> str:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        files.append(path)
        return path

    yield _make
    for p in files:
        if os.path.exists(p):
            os.unlink(p)
        for suffix in ("_cleaned.jsonl", "_scored.jsonl"):
            alt = p.replace(".jsonl", suffix)
            if os.path.exists(alt):
                os.unlink(alt)


@pytest.fixture
def seed_jsonl(temp_jsonl):
    return temp_jsonl([
        json.dumps({"instruction": "what is python", "output": "Python is a programming language."}),
        json.dumps({"messages": [{"role": "user", "content": "explain ML"}, {"role": "assistant", "content": "ML is a subset of AI."}]}),
        json.dumps({"instruction": "how to use git", "response": "Use git clone, git commit, git push."}),
    ])


@pytest.fixture
def sample_config_yaml():
    config = {
        "api": {
            "model": "qwen-plus",
            "lang": "en",
            "concurrency": 1,
            "params": {"temperature": 0.8, "max_tokens": 2048},
            "retry": {"max_retries": 3},
        },
        "strategies": [
            {"type": "topic_driven", "weight": 0.5, "topics": [{"topic": "Python programming", "weight": 3}], "total_count": 10},
            {"type": "seed_driven", "weight": 0.3, "seed_file": "./seeds.jsonl", "example_num": 2, "target_count": 5},
            {"type": "self_instruct", "weight": 0.2, "target_count": 5},
        ],
        "quality": {"instruction_min_len": 5, "instruction_max_len": 2000, "output_min_len": 10, "output_max_len": 6000, "dedup": True},
        "cleaner": {"remove_html": True, "remove_urls": True, "remove_emails": True, "dedup": True},
        "output": {"path": "./test_output.jsonl", "format": "alpaca"},
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    yield path
    if os.path.exists(path):
        os.unlink(path)
