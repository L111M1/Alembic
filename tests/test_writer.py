import json
import os

from alembic.config import OutputConfig
from alembic.core.types import GenerationSample
from alembic.writers.jsonl_writer import JSONLWriter


class TestJSONLWriter:
    def test_writes_alpaca_format(self, tmpdir):
        out_path = tmpdir.join("output.jsonl")
        cfg = OutputConfig(path=str(out_path), format="alpaca")
        writer = JSONLWriter(cfg)
        writer.write(GenerationSample(instruction="hello", output="world"))
        writer.close()

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert data["instruction"] == "hello"
        assert data["output"] == "world"

    def test_writes_chatml_format(self, tmpdir):
        out_path = tmpdir.join("output.jsonl")
        cfg = OutputConfig(path=str(out_path), format="chatml")
        writer = JSONLWriter(cfg)
        writer.write(GenerationSample(instruction="hello", output="world", system="be helpful"))
        writer.close()

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert "messages" in data
        assert data["messages"][0]["role"] == "system"
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][2]["role"] == "assistant"

    def test_writes_sharegpt_format(self, tmpdir):
        out_path = tmpdir.join("output.jsonl")
        cfg = OutputConfig(path=str(out_path), format="sharegpt")
        writer = JSONLWriter(cfg)
        writer.write(GenerationSample(instruction="hello", output="world"))
        writer.close()

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert "conversations" in data
        assert data["conversations"][0]["from"] == "human"
        assert data["conversations"][1]["from"] == "gpt"

    def test_writes_multi_turn_messages(self, tmpdir):
        out_path = tmpdir.join("output.jsonl")
        cfg = OutputConfig(path=str(out_path), format="alpaca")
        writer = JSONLWriter(cfg)
        writer.write(GenerationSample(messages=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]))
        writer.close()

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_writes_multi_turn_sharegpt(self, tmpdir):
        out_path = tmpdir.join("output.jsonl")
        cfg = OutputConfig(path=str(out_path), format="sharegpt")
        writer = JSONLWriter(cfg)
        writer.write(GenerationSample(messages=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]))
        writer.close()

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert "conversations" in data
        assert data["conversations"][0]["from"] == "human"
        assert data["conversations"][1]["from"] == "gpt"

    def test_metadata_preserved_in_multi_turn(self, tmpdir):
        out_path = tmpdir.join("output.jsonl")
        cfg = OutputConfig(path=str(out_path), format="alpaca")
        writer = JSONLWriter(cfg)
        writer.write(GenerationSample(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            metadata={"strategy": "topic_driven"},
        ))
        writer.close()

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert data["metadata"]["strategy"] == "topic_driven"
