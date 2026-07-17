import json

from alembic.cleaner import DatasetCleaner
from alembic.cleaner.ops import clean_text
from alembic.config import CleanerConfig


class TestCleaner:
    def test_clean_text_normalizes_invisible_and_control_characters(self):
        assert clean_text("\ufeffＡ\u200bB\x00\r\n\r\n\r\nC 👩‍💻") == "AB\n\nC 👩‍💻"

    def test_clean_file(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "what is python", "output": "Python is a high-level programming language used for web development and data science."}),
            json.dumps({"instruction": "<html>test html</html>", "output": "visit http://example.com for info"}),
            json.dumps({"instruction": "", "output": ""}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(output_min_len=10, output_max_len=8000, minhash_dedup=False)
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

        cfg = CleanerConfig(output_min_len=15, minhash_dedup=False)
        cleaner = DatasetCleaner(cfg)
        kept, dropped = cleaner.clean_file(path, out)
        assert kept == 1
        assert dropped == 1

    def test_clean_mixed_single_and_multi(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "what is python", "output": "Python is a programming language."}),
            json.dumps({"messages": [
                {"role": "user", "content": "Explain ML."},
                {"role": "assistant", "content": "Machine Learning is a subset of artificial intelligence."},
            ]}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(output_min_len=10, minhash_dedup=False)
        cleaner = DatasetCleaner(cfg)
        kept, _ = cleaner.clean_file(path, out)
        assert kept == 2

        with open(out, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        assert results[0]["instruction"] == "what is python"
        assert "messages" in results[1]

    def test_clean_rejects_low_ngram_diversity(self, temp_jsonl):
        """Repetitive output with low char n-gram diversity should be dropped."""
        path = temp_jsonl([
            json.dumps({"instruction": "tell me about cats", "output": " ".join(["the cat sat"] * 30)}),
            json.dumps({"instruction": "explain python", "output": "Python is a versatile high-level programming language with clean syntax."}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(
            output_min_len=10,
            minhash_dedup=False,
            min_ngram_diversity=0.3,
            ngram_diversity_n=3,
            ngram_diversity_unit="char",
        )
        cleaner = DatasetCleaner(cfg)
        kept, dropped = cleaner.clean_file(path, out)
        assert kept == 1
        assert dropped == 1

    def test_clean_keeps_high_ngram_diversity(self, temp_jsonl):
        """Diverse output should pass even with a strict-ish threshold."""
        path = temp_jsonl([
            json.dumps({"instruction": "explain recursion", "output": "Recursion is a method where a function calls itself to solve smaller instances of a problem until reaching a base case."}),
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(
            output_min_len=10,
            minhash_dedup=False,
            min_ngram_diversity=0.5,
        )
        cleaner = DatasetCleaner(cfg)
        kept, _ = cleaner.clean_file(path, out)
        assert kept == 1
