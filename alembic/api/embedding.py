import logging
import os
from typing import Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class EmbeddingClient:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._model = model
        key = api_key or os.environ.get("EMBEDDING_API_KEY") or os.environ.get("API_KEY", "")
        url = base_url or os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("BASE_URL", "") or None
        self._client = OpenAI(api_key=key, base_url=url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [e.embedding for e in response.data]
