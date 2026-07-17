import abc
import json

from alembic.core.types import GenerationSample


class ResponseParser(abc.ABC):
    """Parses raw LLM response text into structured GenerationSample objects.

    Each strategy can configure its own parser, or use the default
    :class:`JSONResponseParser`.  Custom parsers enable format-specific
    handling (ChatML, ShareGPT, etc.) without polluting the strategy.
    """

    @abc.abstractmethod
    def parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        ...


class JSONResponseParser(ResponseParser):
    """Default parser: one JSON object (or list thereof) with instruction/output."""

    def parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        text = self._clean_fences(response_text)
        data = json.loads(text)

        if isinstance(data, list):
            return self._parse_list(data, metadata)
        return self._parse_single(data, metadata)

    @staticmethod
    def _clean_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _parse_list(data: list, metadata: dict = None) -> list[GenerationSample]:
        results: list[GenerationSample] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            sample = _item_to_sample(item, metadata)
            if sample is not None:
                results.append(sample)
        return results

    @staticmethod
    def _parse_single(data: dict, metadata: dict = None) -> list[GenerationSample]:
        sample = _item_to_sample(data, metadata)
        if sample is None:
            return []
        return [sample]


class InstructionBacktranslationParser(ResponseParser):
    """Parse a reverse-generated instruction while locking the source output.

    The source text is passed through the private ``_source_text`` metadata
    field. Any ``output`` or ``system`` returned by the model is deliberately
    ignored so the generated pair cannot drift away from the human-written
    source document.
    """

    _ALLOWED_TASK_TYPES = {
        "qa",
        "explanation",
        "analysis",
        "how_to",
        "comparison",
        "writing",
        "other",
    }

    def parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        text = JSONResponseParser._clean_fences(response_text)
        data = json.loads(text)
        items = self._extract_items(data)
        source_text = str((metadata or {}).get("_source_text", "")).strip()
        if not source_text:
            return []

        public_metadata = {
            key: value
            for key, value in (metadata or {}).items()
            if not key.startswith("_")
        }
        samples: list[GenerationSample] = []
        for item in items:
            instruction = str(item.get("instruction", "")).strip()
            if not instruction:
                continue
            sample_metadata = dict(public_metadata)
            task_type = str(item.get("task_type", "other")).strip().lower()
            if task_type not in self._ALLOWED_TASK_TYPES:
                task_type = "other"
            sample_metadata["task_type"] = task_type
            samples.append(
                GenerationSample(
                    instruction=instruction,
                    output=source_text,
                    metadata=sample_metadata,
                )
            )
        return samples

    @staticmethod
    def _extract_items(data) -> list[dict]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if not isinstance(data, dict):
            return []
        for key in ("items", "candidates", "instructions"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]


class DocumentQAParser(ResponseParser):
    """Parse grounded question-answer pairs and retain source provenance."""

    def parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        text = JSONResponseParser._clean_fences(response_text)
        data = json.loads(text)
        items = InstructionBacktranslationParser._extract_items(data)
        public_metadata = {
            key: value
            for key, value in (metadata or {}).items()
            if not key.startswith("_")
        }
        samples: list[GenerationSample] = []
        for item in items:
            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()
            if not instruction or not output:
                continue
            sample_metadata = dict(public_metadata)
            task_type = str(item.get("task_type", "qa")).strip().lower()
            if task_type not in InstructionBacktranslationParser._ALLOWED_TASK_TYPES:
                task_type = "other"
            sample_metadata["task_type"] = task_type
            samples.append(
                GenerationSample(
                    instruction=instruction,
                    output=output,
                    metadata=sample_metadata,
                )
            )
        return samples


def _item_to_sample(item: dict, metadata: dict = None) -> GenerationSample | None:
    if "messages" in item:
        sample = GenerationSample(messages=item["messages"])
        if metadata:
            sample.metadata = dict(metadata)
        return sample

    instruction = item.get("instruction", "").strip()
    if not instruction:
        return None

    sample = GenerationSample(
        instruction=instruction,
        output=item.get("output", "").strip(),
        reasoning=item.get("reasoning", "").strip(),
        system=item.get("system", "").strip(),
    )
    if metadata:
        sample.metadata = dict(metadata)
    return sample
