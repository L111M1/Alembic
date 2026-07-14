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
