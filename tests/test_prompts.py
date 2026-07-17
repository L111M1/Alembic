from alembic.prompts.builder import PromptBuilder


class TestPromptBuilder:
    def test_build_basic_messages(self):
        builder = PromptBuilder()
        builder.system("You are helpful")
        builder.user("Hello")
        msgs = builder.build()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"

    def test_from_template(self):
        builder = PromptBuilder()
        builder.from_template("topic_driven_system.j2")
        builder.from_template("topic_driven_user.j2", topic="Python")
        msgs = builder.build()
        assert len(msgs) == 2
        assert "Python" in msgs[1]["content"]

    def test_from_template_zh_fallback(self):
        builder = PromptBuilder(lang="zh")
        builder.from_template("topic_driven_system.j2")
        msgs = builder.build()
        assert len(msgs) == 1

    def test_parse_chat_messages_multirole(self):
        builder = PromptBuilder()
        builder.system("role1")
        builder.user("role2")
        builder.assistant("role3")
        msgs = builder.build()
        assert len(msgs) == 3
        assert msgs[2]["role"] == "assistant"

    def test_multi_turn_template_render(self):
        builder = PromptBuilder()
        builder.from_template("topic_driven_system_mt.j2")
        builder.from_template("topic_driven_user_mt.j2", topic="Python")
        msgs = builder.build()
        assert len(msgs) >= 2
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        assert "messages" in system.lower()
        assert "Python" in user
        assert "messages" in user

    def test_scorer_template_allows_arbitrary_rubric_tiers(self):
        dimensions = [{
            "name": "quality",
            "label": "Quality",
            "max_score": 5,
            "rubric": [
                {"range": str(score), "desc": f"Tier {score}"}
                for score in range(1, 6)
            ],
        }]
        builder = PromptBuilder()
        builder.from_template("scorer_system.j2", dimensions=dimensions)
        system = builder.build()[0]["content"]

        assert "Tier 1" in system
        assert "Tier 5" in system
        assert "4 distinct tiers" not in system
