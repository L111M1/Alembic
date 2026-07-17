import json
import time

import pytest

from alembic.api.base import BaseAPIClient
from alembic.core.parser import InstructionBacktranslationParser
from alembic.registry import strategy_registry
from alembic.strategies.instruction_backtranslation import (
    InstructionBacktranslationStrategy,
)


class DocumentAwareAPI(BaseAPIClient):
    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "alpha source" in user:
            time.sleep(0.02)
            instruction = "What does the alpha source explain?"
        else:
            instruction = "What does the beta source explain?"
        return json.dumps(
            {
                "instruction": instruction,
                "task_type": "explanation",
                "output": "This model-written output must be ignored.",
                "system": "This model-written system must be ignored.",
            }
        )


def _params(document_file, **overrides):
    params = {
        "document_file": document_file,
        "min_document_length": 1,
        "max_document_length": 1000,
    }
    params.update(overrides)
    return params


class TestInstructionBacktranslationParser:
    def test_locks_source_output_and_removes_private_metadata(self):
        parser = InstructionBacktranslationParser()
        samples = parser.parse(
            json.dumps(
                {
                    "instruction": "Why are generators memory efficient?",
                    "task_type": "explanation",
                    "output": "fabricated",
                    "system": "fabricated",
                }
            ),
            {
                "strategy": "instruction_backtranslation",
                "source_id": "doc-1",
                "_source_text": "Generators produce values lazily.",
            },
        )

        assert len(samples) == 1
        sample = samples[0]
        assert sample.output == "Generators produce values lazily."
        assert sample.system == ""
        assert sample.metadata["source_id"] == "doc-1"
        assert sample.metadata["task_type"] == "explanation"
        assert "_source_text" not in sample.metadata

    def test_supports_wrapped_candidates_and_normalizes_unknown_type(self):
        parser = InstructionBacktranslationParser()
        samples = parser.parse(
            json.dumps(
                {
                    "candidates": [
                        {"instruction": "Explain lazy evaluation.", "task_type": "unknown"},
                        {"instruction": ""},
                    ]
                }
            ),
            {"_source_text": "Lazy evaluation delays computation."},
        )

        assert len(samples) == 1
        assert samples[0].metadata["task_type"] == "other"

    def test_requires_source_text(self):
        parser = InstructionBacktranslationParser()
        assert parser.parse('{"instruction": "Explain it."}', {}) == []


class TestInstructionBacktranslationStrategy:
    def test_requires_document_file(self, fake_api):
        with pytest.raises(ValueError, match="document_file"):
            InstructionBacktranslationStrategy(fake_api, {})

    def test_rejects_file_without_valid_documents(self, fake_api, temp_jsonl):
        path = temp_jsonl([json.dumps({"text": "short"})])
        with pytest.raises(ValueError, match="No valid documents"):
            InstructionBacktranslationStrategy(fake_api, {"document_file": path})

    def test_field_mapping_limits_and_metadata(self, fake_api, temp_jsonl):
        path = temp_jsonl(
            [
                json.dumps(
                    {
                        "content": "First complete source response.",
                        "document_id": "a",
                        "origin": "manual",
                        "title": "First",
                    }
                ),
                json.dumps(
                    {
                        "content": "Second complete source response.",
                        "document_id": "b",
                    }
                ),
            ]
        )
        strategy = InstructionBacktranslationStrategy(
            fake_api,
            _params(
                path,
                target_count=1,
                field_map={
                    "content": "text",
                    "document_id": "id",
                    "origin": "source",
                },
            ),
        )

        prompts = list(strategy.iter_prompts())
        assert len(prompts) == 1
        prompt_id, messages = prompts[0]
        assert "First complete source response." in messages[-1]["content"]
        metadata = strategy._build_metadata(prompt_id)
        assert metadata["source_id"] == "a"
        assert metadata["source"] == "manual"
        assert metadata["title"] == "First"
        assert strategy.estimated_count() == 1

    def test_total_count_supports_cli_override(self, fake_api, temp_jsonl):
        path = temp_jsonl(
            [
                json.dumps({"text": "First complete source response."}),
                json.dumps({"text": "Second complete source response."}),
            ]
        )
        strategy = InstructionBacktranslationStrategy(
            fake_api,
            _params(path, target_count=2, total_count=1),
        )

        assert strategy.estimated_count() == 1

    def test_chinese_template(self, fake_api, temp_jsonl):
        path = temp_jsonl([json.dumps({"text": "这是一段可以独立作为回答的完整中文内容。"})])
        strategy = InstructionBacktranslationStrategy(
            fake_api, _params(path, lang="zh")
        )
        _, messages = next(strategy.iter_prompts())

        assert "反推出" in messages[0]["content"]
        assert "必须使用中文" in messages[0]["content"]

    @pytest.mark.parametrize(
        "instruction",
        [
            "根据上文解释生成器。",
            "Read the provided document and summarize it.",
        ],
    )
    def test_rejects_context_dependent_instruction(
        self, fake_api, temp_jsonl, instruction
    ):
        path = temp_jsonl([json.dumps({"text": "A complete source response."})])
        strategy = InstructionBacktranslationStrategy(fake_api, _params(path))
        samples = strategy._parse(
            json.dumps({"instruction": instruction}),
            {"_source_text": "A complete source response."},
        )
        assert samples == []

    def test_can_disable_context_reference_filter(self, fake_api, temp_jsonl):
        path = temp_jsonl([json.dumps({"text": "A complete source response."})])
        strategy = InstructionBacktranslationStrategy(
            fake_api,
            _params(path, reject_context_references=False),
        )
        samples = strategy._parse(
            json.dumps({"instruction": "Summarize the provided document."}),
            {"_source_text": "A complete source response."},
        )
        assert len(samples) == 1

    def test_parallel_generation_keeps_document_pairing(self, temp_jsonl):
        alpha = "This alpha source explains alpha source behavior."
        beta = "This beta source explains beta source behavior."
        path = temp_jsonl(
            [json.dumps({"id": "alpha", "text": alpha}), json.dumps({"id": "beta", "text": beta})]
        )
        strategy = InstructionBacktranslationStrategy(
            DocumentAwareAPI(), _params(path, concurrency=2)
        )

        samples = list(strategy.generate())
        by_id = {sample.metadata["source_id"]: sample for sample in samples}
        assert by_id["alpha"].output == alpha
        assert "alpha" in by_id["alpha"].instruction
        assert by_id["beta"].output == beta
        assert "beta" in by_id["beta"].instruction

    def test_strategy_is_registered(self):
        assert (
            strategy_registry.get("instruction_backtranslation")
            is InstructionBacktranslationStrategy
        )
