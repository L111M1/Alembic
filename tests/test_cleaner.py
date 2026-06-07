from alembic.cleaner import DatasetCleaner
from alembic.config import CleanerConfig


class TestCleaner:
    def test_clean_file(self, temp_jsonl):
        path = temp_jsonl([
            '{"instruction": "what is python", "output": "Python is a high-level programming language used for web development and data science."}',
            '{"instruction": "<html>test html</html>", "output": "visit http://example.com for info"}',
            '{"instruction": "", "output": ""}',
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(remove_html=True, remove_urls=True, output_min_len=10, output_max_len=8000, dedup=False)
        cleaner = DatasetCleaner(cfg)
        kept, dropped = cleaner.clean_file(path, out)
        assert kept == 2
        assert dropped == 1

    def test_clean_file_with_response_field(self, temp_jsonl):
        path = temp_jsonl([
            '{"instruction": "some question", "response": "some answer that is long enough to pass min length"}',
        ])
        out = path.replace(".jsonl", "_cleaned.jsonl")

        cfg = CleanerConfig(instruction_min_len=3, output_min_len=10)
        cleaner = DatasetCleaner(cfg)
        kept, _ = cleaner.clean_file(path, out)
        assert kept == 1
