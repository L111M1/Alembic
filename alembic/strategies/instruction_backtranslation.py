import json
import logging
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator

from alembic.api.base import BaseAPIClient
from alembic.core.parser import InstructionBacktranslationParser
from alembic.core.types import GenerationSample
from alembic.prompts.builder import PromptBuilder
from alembic.strategies.base import GenerationStrategy

logger = logging.getLogger(__name__)


class InstructionBacktranslationStrategy(GenerationStrategy):
    """Infer a user instruction for each human-written source response."""

    _CONTEXT_REFERENCE_RE = re.compile(
        r"(?:根据(?:上文|原文|这篇文章|上述内容|给定材料)|"
        r"阅读(?:上文|原文|以下材料|这篇文章)|"
        r"上述(?:文章|文本|内容|材料)|"
        r"\b(?:the\s+)?(?:text|passage|document|article)\s+(?:above|provided|given)\b|"
        r"\b(?:given|provided)\s+(?:text|passage|document|article)\b)",
        re.IGNORECASE,
    )

    def __init__(self, api: BaseAPIClient, params: dict):
        super().__init__(api, params)
        document_file = params.get("document_file")
        if not document_file:
            raise ValueError("instruction_backtranslation requires document_file")

        self._document_file = Path(document_file)
        if not self._document_file.is_file():
            raise ValueError(f"Document file does not exist: {self._document_file}")

        self._field_map = params.get("field_map") or {}
        # ``total_count`` is injected by the CLI's global --count override.
        self._target_count = max(
            0, int(params.get("total_count", params.get("target_count", 0)))
        )
        self._min_document_length = max(1, int(params.get("min_document_length", 50)))
        self._max_document_length = max(
            self._min_document_length,
            int(params.get("max_document_length", 4000)),
        )
        self._max_instruction_length = max(
            1, int(params.get("max_instruction_length", 500))
        )
        self._reject_context_references = bool(
            params.get("reject_context_references", True)
        )
        self._shuffle = bool(params.get("shuffle", False))
        self._documents = self._load_documents()
        if not self._documents:
            raise ValueError("No valid documents found for instruction backtranslation")
        if self._shuffle:
            random.shuffle(self._documents)
        if self._target_count > 0:
            self._documents = self._documents[: self._target_count]
        self._metadata_lookup: dict[str, dict] = {}
        self._parser = InstructionBacktranslationParser()

    def _load_documents(self) -> list[dict]:
        documents: list[dict] = []
        with self._document_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping invalid JSON in %s at line %d",
                        self._document_file,
                        line_number,
                    )
                    continue
                if not isinstance(raw, dict):
                    continue
                item = dict(raw)
                for source_field, canonical_field in self._field_map.items():
                    item[canonical_field] = raw.get(source_field, "")
                text = str(item.get("text", "")).strip()
                if not (
                    self._min_document_length
                    <= len(text)
                    <= self._max_document_length
                ):
                    continue
                documents.append(
                    {
                        "text": text,
                        "id": str(item.get("id", line_number)),
                        "source": item.get("source", ""),
                        "title": item.get("title", ""),
                        "metadata": item.get("metadata", {}),
                        "line_number": line_number,
                    }
                )
        return documents

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        for index, document in enumerate(self._documents):
            prompt_id = f"instruction_backtranslation:{index}"
            metadata = {
                "strategy": "instruction_backtranslation",
                "source_id": document["id"],
                "source_index": document["line_number"],
                "_source_text": document["text"],
            }
            if document["source"]:
                metadata["source"] = document["source"]
            if document["title"]:
                metadata["title"] = document["title"]
            if isinstance(document["metadata"], dict) and document["metadata"]:
                metadata["source_metadata"] = document["metadata"]
            self._metadata_lookup[prompt_id] = metadata

            builder = PromptBuilder(lang=self._lang)
            builder.from_template("backtranslation_system.j2")
            builder.from_template(
                "backtranslation_user.j2",
                document=document["text"],
                title=document["title"],
            )
            yield prompt_id, builder.build()

    def _build_metadata(self, prompt_id: str) -> dict:
        return dict(self._metadata_lookup.get(prompt_id, {}))

    def _parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        samples = self._parser.parse(response_text=response_text, metadata=metadata)
        return [sample for sample in samples if self._is_valid_instruction(sample)]

    def _is_valid_instruction(self, sample: GenerationSample) -> bool:
        instruction = sample.instruction.strip()
        if not instruction or len(instruction) > self._max_instruction_length:
            return False
        if self._reject_context_references and self._CONTEXT_REFERENCE_RE.search(instruction):
            return False
        source = sample.output.strip()
        if instruction == source:
            return False
        if len(instruction) >= 30:
            copied_ratio = SequenceMatcher(None, instruction, source).find_longest_match(
                0, len(instruction), 0, len(source)
            ).size / len(instruction)
            if copied_ratio >= 0.8:
                return False
        return True

    def estimated_count(self) -> int:
        return len(self._documents)
