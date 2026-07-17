import json

from alembic.documents.chunker import DocumentChunker
from alembic.documents.loader import load_document_chunks


class FakeEmbeddingClient:
    def embed(self, texts):
        vectors = {
            "Alpha topic first paragraph.": [1.0, 0.0],
            "Alpha topic second paragraph.": [0.99, 0.01],
            "Beta topic unrelated paragraph.": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


class TestDocumentChunker:
    def test_structure_chunking_respects_max_length(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = DocumentChunker(
            mode="structure", min_chunk_length=10, max_chunk_length=30
        )
        chunks = chunker.chunk(text)

        assert len(chunks) >= 2
        assert all(len(chunk) <= 30 for chunk in chunks)

    def test_semantic_chunking_breaks_on_topic_shift(self):
        text = (
            "Alpha topic first paragraph.\n\n"
            "Alpha topic second paragraph.\n\n"
            "Beta topic unrelated paragraph."
        )
        chunker = DocumentChunker(
            mode="semantic",
            min_chunk_length=20,
            max_chunk_length=200,
            similarity_threshold=0.8,
            embedding_client=FakeEmbeddingClient(),
        )
        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        assert "first paragraph" in chunks[0]
        assert "second paragraph" in chunks[0]
        assert chunks[1] == "Beta topic unrelated paragraph."


class TestDocumentLoader:
    def test_normalizes_special_characters_before_chunking(self, tmp_path):
        path = tmp_path / "noisy.md"
        path.write_text(
            "\ufeff＃ 标题\r\n\r\n正文\u200b包含\u00a0空格\x00和 emoji 👩‍💻。\r\n\r\n\r\n结尾",
            encoding="utf-8",
        )

        chunks = load_document_chunks(str(path))

        assert len(chunks) == 1
        assert chunks[0]["text"] == "# 标题\n\n正文包含 空格和 emoji 👩‍💻。\n\n结尾"

    def test_loads_and_chunks_markdown(self, tmp_path):
        path = tmp_path / "guide.md"
        path.write_text(
            "# First\n\nFirst section content about Python.\n\n"
            "# Second\n\nSecond section content about databases.",
            encoding="utf-8",
        )
        chunks = load_document_chunks(
            str(path),
            chunking={
                "enabled": True,
                "mode": "structure",
                "min_chunk_length": 10,
                "max_chunk_length": 55,
            },
        )

        assert len(chunks) >= 2
        assert all(item["source_id"] == "guide" for item in chunks)
        assert [item["chunk_index"] for item in chunks] == list(range(len(chunks)))
        assert all(item["chunk_count"] == len(chunks) for item in chunks)

    def test_loads_json_array_with_field_mapping(self, tmp_path):
        path = tmp_path / "docs.json"
        path.write_text(
            json.dumps([{"doc_id": "a", "content": "Mapped document text."}]),
            encoding="utf-8",
        )
        chunks = load_document_chunks(
            str(path), field_map={"doc_id": "id", "content": "text"}
        )

        assert len(chunks) == 1
        assert chunks[0]["id"] == "a"
        assert chunks[0]["text"] == "Mapped document text."
