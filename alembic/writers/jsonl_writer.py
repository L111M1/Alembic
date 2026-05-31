import abc
import json
import logging
import os
from pathlib import Path
from typing import Optional

from alembic.core.types import GenerationSample
from alembic.config import OutputConfig

logger = logging.getLogger(__name__)


class BaseWriter(abc.ABC):
    @abc.abstractmethod
    def write(self, sample: GenerationSample) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def count(self) -> int: ...


class JSONLWriter(BaseWriter):
    def __init__(self, config: OutputConfig):
        self._path = Path(config.path)
        self._format = config.format
        self._count = 0
        self._file = None
        self._checkpoint_path = config.checkpoint_path

        os.makedirs(self._path.parent, exist_ok=True)

    def write(self, sample: GenerationSample) -> None:
        if self._file is None:
            exists = self._path.exists()
            self._file = open(self._path, "a", encoding="utf-8")
            if exists:
                self._file.write("\n")

        record = self._format_sample(sample)
        line = json.dumps(record, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        self._count += 1

        if self._checkpoint_path and self._count % 10 == 0:
            self._save_checkpoint()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        if self._checkpoint_path:
            self._save_checkpoint()

    @property
    def count(self) -> int:
        return self._count

    def _format_sample(self, sample: GenerationSample) -> dict:
        if self._format == "chatml":
            messages = []
            if sample.system:
                messages.append({"role": "system", "content": sample.system})
            messages.append({"role": "user", "content": sample.instruction})
            messages.append({"role": "assistant", "content": sample.output})
            return {"messages": messages}
        elif self._format == "sharegpt":
            conversations = []
            if sample.system:
                conversations.append({"from": "system", "value": sample.system})
            conversations.append({"from": "human", "value": sample.instruction})
            conversations.append({"from": "gpt", "value": sample.output})
            return {"conversations": conversations}
        else:
            record = {"instruction": sample.instruction, "output": sample.output}
            if sample.system:
                record["system"] = sample.system
            return record

    def _save_checkpoint(self) -> None:
        if self._checkpoint_path:
            with open(self._checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"count": self._count}, f)
