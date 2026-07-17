import re
from typing import Optional

from alembic.api.embedding import EmbeddingClient, cosine_similarity


class DocumentChunker:
    """Split documents structurally, with optional adjacent semantic boundaries."""

    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？；;])\s+")

    def __init__(
        self,
        mode: str = "structure",
        min_chunk_length: int = 200,
        max_chunk_length: int = 1500,
        similarity_threshold: float = 0.55,
        embedding_model: str = "text-embedding-v3",
        embedding_batch_size: int = 32,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        if mode not in {"structure", "semantic"}:
            raise ValueError("chunking mode must be 'structure' or 'semantic'")
        self._mode = mode
        self._min_length = max(1, int(min_chunk_length))
        self._max_length = max(self._min_length, int(max_chunk_length))
        self._threshold = float(similarity_threshold)
        self._embedding_model = embedding_model
        self._batch_size = max(1, int(embedding_batch_size))
        self._embedding_client = embedding_client

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        segments = self._split_atomic(text)
        if len(segments) <= 1:
            return segments

        similarities = None
        if self._mode == "semantic":
            similarities = self._adjacent_similarities(segments)
        chunks = self._merge_segments(segments, similarities)
        return self._merge_short_tail(chunks)

    def _split_atomic(self, text: str) -> list[str]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
        segments: list[str] = []
        for block in blocks:
            if len(block) <= self._max_length:
                segments.append(block)
                continue
            sentences = [
                sentence.strip()
                for sentence in self._SENTENCE_BOUNDARY.split(block)
                if sentence.strip()
            ]
            if len(sentences) == 1:
                segments.extend(
                    block[start:start + self._max_length].strip()
                    for start in range(0, len(block), self._max_length)
                )
            else:
                segments.extend(self._pack_sentences(sentences))
        return [segment for segment in segments if segment]

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        packed: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > self._max_length:
                if current:
                    packed.append(current)
                    current = ""
                packed.extend(
                    sentence[start:start + self._max_length].strip()
                    for start in range(0, len(sentence), self._max_length)
                )
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > self._max_length:
                packed.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            packed.append(current)
        return packed

    def _adjacent_similarities(self, segments: list[str]) -> list[float]:
        client = self._embedding_client or EmbeddingClient(model=self._embedding_model)
        vectors: list[list[float]] = []
        for start in range(0, len(segments), self._batch_size):
            vectors.extend(client.embed(segments[start:start + self._batch_size]))
        if len(vectors) != len(segments):
            raise ValueError("Embedding service returned an unexpected vector count")
        return [
            cosine_similarity(vectors[index], vectors[index + 1])
            for index in range(len(vectors) - 1)
        ]

    def _merge_segments(
        self, segments: list[str], similarities: Optional[list[float]]
    ) -> list[str]:
        chunks: list[str] = []
        current = segments[0]
        for index, segment in enumerate(segments[1:]):
            candidate = f"{current}\n\n{segment}"
            over_limit = len(candidate) > self._max_length
            semantic_break = (
                similarities is not None
                and len(current) >= self._min_length
                and similarities[index] < self._threshold
            )
            if over_limit or semantic_break:
                chunks.append(current)
                current = segment
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _merge_short_tail(self, chunks: list[str]) -> list[str]:
        if len(chunks) < 2 or len(chunks[-1]) >= self._min_length:
            return chunks
        combined = f"{chunks[-2]}\n\n{chunks[-1]}"
        if len(combined) <= self._max_length:
            return [*chunks[:-2], combined]
        return chunks

