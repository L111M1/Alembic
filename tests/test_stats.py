import json
import os

from alembic.core.stats import StatisticsCollector
from alembic.core.types import GenerationStats
from collections import Counter


class TestStatisticsCollector:
    def test_collects_generation_stats(self):
        collector = StatisticsCollector()
        collector.on_start(10)

        collector.record_sample({"instruction": "hello", "output": "world", "metadata": {"strategy": "topic_driven", "topic": "Python"}})
        collector.record_sample({"instruction": "longer question", "output": "detailed answer here", "metadata": {"strategy": "seed_driven"}})

        stats = GenerationStats(total_generated=2, total_filtered=1, by_strategy=Counter({"topic_driven": 1, "seed_driven": 1}))
        collector.on_complete(stats)

        report = collector.generate_report()
        assert report["pipeline"]["total_generated"] == 2
        assert report["pipeline"]["total_filtered"] == 1
        assert report["by_strategy"] == {"topic_driven": 1, "seed_driven": 1}
        assert report["by_topic"] == {"Python": 1}
        assert report["length_distribution"]["instruction_length"]["count"] == 2
        assert report["length_distribution"]["instruction_length"]["min"] == len("hello")
        assert report["length_distribution"]["instruction_length"]["max"] == len("longer question")

    def test_collects_cleaner_stats(self):
        collector = StatisticsCollector()
        collector.on_start(10)
        collector.record_cleaner(8, 2)
        collector.on_complete(GenerationStats())

        report = collector.generate_report()
        assert report["cleaner"]["kept"] == 8
        assert report["cleaner"]["dropped"] == 2
        assert report["cleaner"]["retention_rate"] == 0.8

    def test_collects_scorer_stats(self, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "x", "output": "y", "scores": {"correctness": 8, "helpfulness": 7}, "total_score": 15}),
            json.dumps({"instruction": "a", "output": "b", "scores": {"correctness": 6, "helpfulness": 5}, "total_score": 11}),
        ])

        collector = StatisticsCollector()
        collector.on_start(2)
        collector.record_cleaner(2, 0)
        collector.record_scorer(2, 0)
        collector.record_scores(path)
        collector.record_score_filter(1, 1)
        collector.on_complete(GenerationStats())

        report = collector.generate_report()
        assert report["scorer"]["scored"] == 2
        assert report["scorer"]["failed"] == 0
        assert report["scorer"]["total_score_distribution"]["count"] == 2
        assert report["scorer"]["total_score_distribution"]["mean"] == 13.0
        assert "correctness" in report["scorer"]["dimension_distributions"]
        assert report["scorer"]["score_filter"]["kept"] == 1
        assert report["scorer"]["score_filter"]["dropped"] == 1

    def test_collects_multi_turn_stats(self):
        collector = StatisticsCollector()
        collector.on_start(5)

        collector.record_sample({
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "A programming language."},
            ],
            "metadata": {"strategy": "topic_driven", "topic": "Python"},
        })

        collector.on_complete(GenerationStats(total_generated=1))

        report = collector.generate_report()
        ld = report["length_distribution"]
        assert ld["instruction_length"]["count"] == 1
        assert ld["output_length"]["count"] == 1

    def test_saves_report_to_file(self, temp_jsonl, tmpdir):
        collector = StatisticsCollector()
        collector.on_start(5)
        collector.record_sample({"instruction": "test", "output": "value"})
        collector.on_complete(GenerationStats(total_generated=1))

        out_path = str(tmpdir.join("my_output.jsonl"))
        report_path = collector.save_report(out_path)
        assert os.path.exists(report_path)
        assert report_path.endswith("_report.json")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["pipeline"]["total_generated"] == 1

    def test_empty_stats_report(self):
        collector = StatisticsCollector()
        collector.on_start(0)
        collector.on_complete(GenerationStats())

        report = collector.generate_report()
        assert report["pipeline"]["total_generated"] == 0
        assert report["pipeline"]["pass_rate"] == 0.0
