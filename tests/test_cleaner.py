import json

from alembic.cleaner import DatasetCleaner
from alembic.config import CleanerConfig


class TestCleaner:
    def test_clean_file(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "what is python", "output": "Python is a high-level programming language used for web development and data science."}),
            json.dumps({"instruction": "<html>test html</html>", "output": "visit http://example.com for info"}),
            json.dumps({"instruction": "", "output": ""}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(remove_html=True, remove_urls=True, output_min_len=10, output_max_len=8000, minhash_dedup=False)
        cleaner = DatasetCleaner(cfg)
        kept, dropped = cleaner.clean_file(path, out)
        assert kept == 2
        assert dropped == 1

    def test_clean_file_with_response_field(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "some question", "response": "some answer that is long enough to pass min length"}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(instruction_min_len=3, output_min_len=10, minhash_dedup=False)
        cleaner = DatasetCleaner(cfg)
        kept, _ = cleaner.clean_file(path, out)
        assert kept == 1

    def test_clean_multi_turn(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language for many purposes."},
            ]}),
            json.dumps({"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(remove_html=True, remove_urls=True, output_min_len=15, minhash_dedup=False)
        cleaner = DatasetCleaner(cfg)
        kept, dropped = cleaner.clean_file(path, out)
        assert kept == 1
        assert dropped == 1

    def test_clean_multi_turn_html_removal(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"messages": [
                {"role": "user", "content": "<b>What is Python?</b>"},
                {"role": "assistant", "content": "Python is a language."},
            ]}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(remove_html=True, remove_urls=True, output_min_len=5, minhash_dedup=False)
        cleaner = DatasetCleaner(cfg)
        kept, _ = cleaner.clean_file(path, out)
        assert kept == 1

        with open(out, "r", encoding="utf-8") as f:
            cleaned = json.loads(f.readline())
        assert "<b>" not in cleaned["messages"][0]["content"]

    def test_clean_mixed_single_and_multi(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "what is python", "output": "Python is a programming language."}),
            json.dumps({"messages": [
                {"role": "user", "content": "Explain ML."},
                {"role": "assistant", "content": "Machine Learning is a subset of artificial intelligence."},
            ]}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(remove_html=True, remove_urls=True, output_min_len=10, minhash_dedup=False)
        cleaner = DatasetCleaner(cfg)
        kept, _ = cleaner.clean_file(path, out)
        assert kept == 2

        with open(out, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        assert results[0]["instruction"] == "what is python"
        assert "messages" in results[1]
