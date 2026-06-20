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


class FakeMultiTurnAPI(BaseAPIClient):
    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        return json.dumps({
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "What can I build with it?"},
                {"role": "assistant", "content": "You can build web apps, ML models, and more."},
            ]
        })


class FakeBatchAPI(BaseAPIClient):
    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        import re
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        match = re.search(r"Generate (\d+) diverse", user_msg)
        count = int(match.group(1)) if match else 1
        return json.dumps([
            {"instruction": f"Q{i}: test question", "output": f"A{i}: test answer"}
            for i in range(count)
        ])


class FakeBatchMultiTurnAPI(BaseAPIClient):
    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        import re
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        match = re.search(r"Generate (\d+) diverse", user_msg)
        count = int(match.group(1)) if match else 1
        return json.dumps([
            {"messages": [
                {"role": "user", "content": f"Q{i}a"},
                {"role": "assistant", "content": f"A{i}a"},
                {"role": "user", "content": f"Q{i}b"},
                {"role": "assistant", "content": f"A{i}b"},
            ]}
            for i in range(count)
        ])


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
def fake_multi_turn_api():
    return FakeMultiTurnAPI()


@pytest.fixture
def fake_batch_api():
    return FakeBatchAPI()


@pytest.fixture
def fake_batch_multi_turn_api():
    return FakeBatchMultiTurnAPI()


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
        for suffix in ("_cleaned.jsonl", "_scored.jsonl", "_scored_filtered.jsonl", "_report.json"):
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
def multi_turn_seed_jsonl(temp_jsonl):
    return temp_jsonl([
        json.dumps({"messages": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Is it good for web dev?"},
            {"role": "assistant", "content": "Yes, with Django or Flask."},
        ]}),
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
        "scoring": {
            "enabled": False,
            "model": "gpt-4o",
            "lang": "zh",
            "concurrency": 3,
            "dimensions": [
                {"name": "correctness", "label": "准确性", "max_score": 10},
                {"name": "helpfulness", "label": "实用性", "max_score": 10},
                {"name": "completeness", "label": "完整性", "max_score": 10},
                {"name": "clarity", "label": "清晰度", "max_score": 10},
            ],
        },
        "quality": {"instruction_min_len": 5, "instruction_max_len": 2000, "output_min_len": 10, "output_max_len": 6000, "dedup": True},
        "cleaner": {"remove_html": True, "remove_urls": True, "remove_emails": True, "dedup": True},
        "output": {"path": "./test_output.jsonl", "format": "alpaca", "multi_turn": False},
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    yield path
    if os.path.exists(path):
        os.unlink(path)
