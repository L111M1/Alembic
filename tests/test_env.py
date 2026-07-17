import os

from alembic.env import load_environment


def test_dotenv_overrides_process_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_KEY", "from-process")
    (tmp_path / ".env").write_text("API_KEY=from-dotenv\n", encoding="utf-8")

    assert load_environment() is True
    assert os.environ["API_KEY"] == "from-dotenv"


def test_missing_dotenv_key_falls_back_to_process_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BASE_URL", "https://from-process.example/v1")
    (tmp_path / ".env").write_text("API_KEY=from-dotenv\n", encoding="utf-8")

    assert load_environment() is True
    assert os.environ["BASE_URL"] == "https://from-process.example/v1"


def test_missing_dotenv_preserves_process_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_KEY", "from-process")

    assert load_environment() is False
    assert os.environ["API_KEY"] == "from-process"
