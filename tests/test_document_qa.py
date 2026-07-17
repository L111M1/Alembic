import json

from alembic.api.base import BaseAPIClient
from alembic.core.parser import DocumentQAParser
from alembic.registry import strategy_registry
from alembic.strategies.document_qa import DocumentQAStrategy


class FakeDocumentQAAPI(BaseAPIClient):
    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        return json.dumps(
            {
                "instruction": "How does lazy evaluation reduce memory usage?",
                "output": "It produces values only when they are requested.",
                "task_type": "explanation",
            }
        )


class TestDocumentQAParser:
    def test_parses_generated_answer_and_grounding_metadata(self):
        parser = DocumentQAParser()
        samples = parser.parse(
            json.dumps(
                {
                    "instruction": "What does yield do?",
                    "output": "It produces values lazily.",
                    "task_type": "qa",
                }
            ),
            {"source_id": "doc", "source_text": "Yield produces values lazily."},
        )

        assert len(samples) == 1
        assert samples[0].output == "It produces values lazily."
        assert samples[0].metadata["source_text"] == "Yield produces values lazily."


class TestDocumentQAStrategy:
    def test_requires_document_file(self):
        try:
            DocumentQAStrategy(FakeDocumentQAAPI(), {})
        except ValueError as exc:
            assert "document_file" in str(exc)
        else:
            raise AssertionError("Expected missing document_file to fail")

    def test_generates_qa_from_markdown_chunks(self, tmp_path):
        path = tmp_path / "source.md"
        path.write_text(
            "Python generators use yield and produce values lazily, which reduces memory usage.",
            encoding="utf-8",
        )
        strategy = DocumentQAStrategy(
            FakeDocumentQAAPI(),
            {
                "document_file": str(path),
                "lang": "en",
                "chunking": {"enabled": True, "mode": "structure"},
            },
        )
        samples = list(strategy.generate())

        assert len(samples) == 1
        assert samples[0].instruction.startswith("How does")
        assert samples[0].metadata["strategy"] == "document_qa"
        assert "generators use yield" in samples[0].metadata["source_text"]

    def test_strategy_is_registered(self):
        assert strategy_registry.get("document_qa") is DocumentQAStrategy

