import json
from pathlib import Path

from alembic.documents.chunker import DocumentChunker
from alembic.text import normalize_text


def load_document_chunks(
    path: str,
    field_map: dict | None = None,
    chunking: dict | None = None,
    embedding_client=None,
) -> list[dict]:
    """Load JSON/JSONL/TXT/Markdown and return normalized document chunks."""

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Document file does not exist: {source_path}")
    documents = _load_documents(source_path, field_map or {})
    config = chunking or {}
    enabled = bool(config.get("enabled", False))
    chunker = None
    if enabled:
        chunker = DocumentChunker(
            mode=config.get("mode", "structure"),
            min_chunk_length=config.get("min_chunk_length", 200),
            max_chunk_length=config.get("max_chunk_length", 1500),
            similarity_threshold=config.get("similarity_threshold", 0.55),
            embedding_model=config.get("embedding_model", "text-embedding-v3"),
            embedding_batch_size=config.get("embedding_batch_size", 32),
            embedding_client=embedding_client,
        )

    results: list[dict] = []
    for document in documents:
        texts = chunker.chunk(document["text"]) if chunker else [document["text"]]
        count = len(texts)
        for index, text in enumerate(texts):
            item = dict(document)
            item["text"] = text.strip()
            item["source_id"] = document["id"]
            item["id"] = f'{document["id"]}:chunk-{index}' if count > 1 else document["id"]
            item["chunk_index"] = index
            item["chunk_count"] = count
            results.append(item)
    return results


def _load_documents(path: Path, field_map: dict) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return [
            _normalize_document(
                {"id": path.stem, "text": path.read_text(encoding="utf-8"), "title": path.stem},
                path,
                1,
            )
        ]
    if suffix == ".jsonl":
        raw_items = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    raw_items.append((item, line_number))
        return [
            _normalize_document(_apply_field_map(item, field_map), path, line_number)
            for item, line_number in raw_items
        ]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_items = data if isinstance(data, list) else [data]
        return [
            _normalize_document(_apply_field_map(item, field_map), path, index)
            for index, item in enumerate(raw_items, 1)
            if isinstance(item, dict)
        ]
    raise ValueError("Supported document formats: .jsonl, .json, .txt, .md, .markdown")


def _apply_field_map(item: dict, field_map: dict) -> dict:
    mapped = dict(item)
    for source_field, canonical_field in field_map.items():
        mapped[canonical_field] = item.get(source_field, "")
    return mapped


def _normalize_document(item: dict, path: Path, index: int) -> dict:
    text = normalize_text(item.get("text", ""))
    return {
        "id": str(item.get("id", f"{path.stem}-{index}")),
        "text": text,
        "source": item.get("source", str(path)),
        "title": item.get("title", path.stem),
        "metadata": item.get("metadata", {}),
        "source_path": str(path),
        "line_number": index,
    }
