"""Environment variable loading helpers."""

from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv


def load_environment(dotenv_path: Optional[Union[str, Path]] = None) -> bool:
    """Load a local ``.env`` file, overriding matching process variables.

    When no file is supplied, ``.env`` is resolved from the current working
    directory. Missing files are ignored, leaving existing process environment
    variables untouched as the fallback.
    """

    path = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    if not path.is_file():
        return False
    return load_dotenv(dotenv_path=path, override=True)
