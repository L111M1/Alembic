import logging
import random
import re
from typing import Iterator

from alembic.api.base import BaseAPIClient
from alembic.core.parser import DocumentQAParser
from alembic.core.types import GenerationSample
from alembic.documents.loader import load_document_chunks
from alembic.prompts.builder import PromptBuilder
from alembic.strategies.base import GenerationStrategy

logger = logging.getLogger(__name__)


class DocumentQAStrategy(GenerationStrategy):
    """Generate grounded instruction-answer pairs from document chunks."""

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
            raise ValueError("document_qa requires document_file")
        self._max_instruction_length = max(
            1, int(params.get("max_instruction_length", 500))
        )
        self._max_output_length = max(1, int(params.get("max_output_length", 4000)))
        self._reject_context_references = bool(
            params.get("reject_context_references", True)
        )
        self._documents = load_document_chunks(
            document_file,
            field_map=params.get("field_map") or {},
            chunking=params.get("chunking") or {},
        )
        self._documents = [item for item in self._documents if item.get("text", "").strip()]
        if bool(params.get("shuffle", False)):
            random.shuffle(self._documents)
        target_count = max(
            0, int(params.get("total_count", params.get("target_count", 0)))
        )
        if target_count:
            self._documents = self._documents[:target_count]
        if not self._documents:
            raise ValueError("No valid document chunks found for document_qa")
        self._metadata_lookup: dict[str, dict] = {}
        self._parser = DocumentQAParser()

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        for index, document in enumerate(self._documents):
            prompt_id = f"document_qa:{index}"
            metadata = {
                "strategy": "document_qa",
                "source_id": document["source_id"],
                "chunk_id": document["id"],
                "chunk_index": document["chunk_index"],
                "chunk_count": document["chunk_count"],
                "source": document.get("source", ""),
                "title": document.get("title", ""),
                "source_text": document["text"],
            }
            if isinstance(document.get("metadata"), dict) and document["metadata"]:
                metadata["source_metadata"] = document["metadata"]
            self._metadata_lookup[prompt_id] = metadata

            builder = PromptBuilder(lang=self._lang)
            builder.from_template("document_qa_system.j2")
            builder.from_template(
                "document_qa_user.j2",
                document=document["text"],
                title=document.get("title", ""),
            )
            yield prompt_id, builder.build()

    def _build_metadata(self, prompt_id: str) -> dict:
        return dict(self._metadata_lookup.get(prompt_id, {}))

    def _parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        samples = self._parser.parse(response_text=response_text, metadata=metadata)
        return [sample for sample in samples if self._is_valid_sample(sample)]

    def _is_valid_sample(self, sample: GenerationSample) -> bool:
        instruction = sample.instruction.strip()
        output = sample.output.strip()
        if not instruction or not output:
            return False
        if len(instruction) > self._max_instruction_length:
            return False
        if len(output) > self._max_output_length:
            return False
        if self._reject_context_references and self._CONTEXT_REFERENCE_RE.search(instruction):
            return False
        return True

    def estimated_count(self) -> int:
        return len(self._documents)

