import json
import os

from alembic.api.base import BaseAPIClient
from alembic.config import ScoringConfig
from alembic.scoring import DatasetScorer


class TestScoringConfig:
    def test_defaults(self):
        cfg = ScoringConfig()
        assert cfg.lang == "en"
        assert cfg.concurrency == 3
        assert cfg.model == "gpt-4o"
        assert cfg.min_total_score == 0.0

    def test_default_dimensions(self):
        cfg = ScoringConfig()
        assert len(cfg.dimensions) == 4
        names = [d["name"] for d in cfg.dimensions]
        assert names == ["correctness", "helpfulness", "completeness", "clarity"]
        for d in cfg.dimensions:
            assert "max_score" in d
            assert d["max_score"] == 10

    def test_custom_dimensions(self):
        dims = [
            {"name": "accuracy", "label": "准确", "max_score": 5},
            {"name": "fluency", "label": "流畅", "max_score": 5},
        ]
        cfg = ScoringConfig(dimensions=dims)
        assert len(cfg.dimensions) == 2
        assert cfg.dimensions[1]["name"] == "fluency"
        assert cfg.dimensions[1]["max_score"] == 5

    def test_field_map_is_optional(self):
        cfg = ScoringConfig()
        assert cfg.field_map is None

    def test_lang_defaults_to_en(self):
        cfg = ScoringConfig()
        assert cfg.lang == "en"

    def test_lang_zh(self):
        cfg = ScoringConfig(lang="zh")
        assert cfg.lang == "zh"


class TestDatasetScorer:
    def test_score_standard_jsonl(self, fake_score_api, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "what is python", "output": "Python is a programming language."}),
            json.dumps({"instruction": "explain async", "output": "Async enables non-blocking I/O."}),
        ])
        out = path.replace(".jsonl", "_scored.jsonl")

        cfg = ScoringConfig()
        scorer = DatasetScorer(cfg)
        scored, failed = scorer.score_file(fake_score_api, path, out)

        assert scored == 2
        assert failed == 0
        assert os.path.exists(out)

        with open(out, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        assert len(results) == 2

        s1 = results[0]
        assert "scores" in s1
        assert s1["scores"]["correctness"] == 9
        assert s1["scores"]["helpfulness"] == 8
        assert s1["scores"]["completeness"] == 7
        assert s1["scores"]["clarity"] == 8
        assert s1["total_score"] == 32

    def test_score_skips_empty_samples(self, fake_score_api, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "", "output": ""}),
            json.dumps({"instruction": "hello", "output": "world"}),
        ])
        out = path.replace(".jsonl", "_scored.jsonl")

        cfg = ScoringConfig()
        scorer = DatasetScorer(cfg)
        scored, failed = scorer.score_file(fake_score_api, path, out)

        assert scored == 1

    def test_score_with_field_map(self, fake_score_api, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"question": "what is python", "answer": "Python is a high-level language."}),
        ])
        out = path.replace(".jsonl", "_scored.jsonl")

        cfg = ScoringConfig(field_map={"question": "instruction", "answer": "output"})
        scorer = DatasetScorer(cfg)
        scored, _ = scorer.score_file(fake_score_api, path, out)

        assert scored == 1
        with open(out, "r", encoding="utf-8") as f:
            s1 = json.loads(f.readline())
        assert s1["instruction"] == "what is python"
        assert "scores" in s1
        assert s1["total_score"] == 32

    def test_score_with_response_field(self, fake_score_api, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "what is pytest", "response": "A testing framework."}),
        ])
        out = path.replace(".jsonl", "_scored.jsonl")

        cfg = ScoringConfig()
        scorer = DatasetScorer(cfg)
        scored, _ = scorer.score_file(fake_score_api, path, out)

        assert scored == 1

    def test_score_with_lang_zh(self, fake_score_api, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "什么是 Python", "output": "Python 是一种编程语言。"}),
        ])
        out = path.replace(".jsonl", "_scored.jsonl")

        cfg = ScoringConfig(lang="zh")
        scorer = DatasetScorer(cfg)
        scored, _ = scorer.score_file(fake_score_api, path, out)

        assert scored == 1

    def test_score_output_path_default(self, fake_score_api, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "x", "output": "y"}),
        ])
        cfg = ScoringConfig()
        scorer = DatasetScorer(cfg)
        scored, _ = scorer.score_file(fake_score_api, path)

        default_out = path.replace(".jsonl", "_scored.jsonl")
        assert scored == 1
        assert os.path.exists(default_out)

    def test_score_custom_dimensions_in_prompt(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "test", "output": "value"}),
        ])
        out = path.replace(".jsonl", "_scored.jsonl")

        dims = [
            {"name": "accuracy", "label": "准确", "max_score": 5},
            {"name": "fluency", "label": "流畅", "max_score": 5},
        ]

        captured_prompt = {}

        class InspectorAPI(BaseAPIClient):
            def supports_json_mode(self):
                return True

            def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
                captured_prompt["messages"] = messages
                captured_prompt["temp"] = temperature
                return json.dumps({"accuracy": 4, "fluency": 5})

        api = InspectorAPI()
        cfg = ScoringConfig(dimensions=dims)
        scorer = DatasetScorer(cfg)
        scored, _ = scorer.score_file(api, path, out)

        assert scored == 1
        system_msg = captured_prompt["messages"][0]["content"]
        assert "accuracy" in system_msg
        assert "fluency" in system_msg
        assert "准确" in system_msg
        assert "流畅" in system_msg
        assert "分值范围 1-5" in system_msg

        user_msg = captured_prompt["messages"][1]["content"]
        assert '"accuracy"' in user_msg
        assert '"fluency"' in user_msg

        with open(out, "r", encoding="utf-8") as f:
            s1 = json.loads(f.readline())
        assert s1["scores"]["accuracy"] == 4
        assert s1["scores"]["fluency"] == 5
        assert s1["total_score"] == 9

    def test_score_empty_file(self, fake_score_api, temp_jsonl):
        path = temp_jsonl([])
        out = path.replace(".jsonl", "_scored.jsonl")

        cfg = ScoringConfig()
        scorer = DatasetScorer(cfg)
        scored, failed = scorer.score_file(fake_score_api, path, out)

        assert scored == 0
        assert failed == 0

    def test_score_no_valid_samples(self, fake_score_api, temp_jsonl):
        path = temp_jsonl([
            '{"instruction": "", "output": ""}',
            'not valid json at all',
            '{"instruction": "only_inst_no_output"}',
            '{"output": "only_output_no_instruction"}',
            '',
        ])
        out = path.replace(".jsonl", "_scored.jsonl")

        cfg = ScoringConfig()
        scorer = DatasetScorer(cfg)
        scored, _ = scorer.score_file(fake_score_api, path, out)

        assert scored == 0
