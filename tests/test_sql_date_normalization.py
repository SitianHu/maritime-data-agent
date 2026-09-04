import asyncio
import json
import unittest
from unittest.mock import patch

from app import llm


class SqlDateNormalizationTests(unittest.TestCase):
    def setUp(self):
        llm.clear_sql_cache()

    def test_normalizes_qualified_exact_date_filter_to_like(self):
        sql = "SELECT * FROM violation_record l WHERE date(l.violation_time) = '2026-07-24'"

        normalized = llm._normalize_exact_date_filters(sql, {"violation_time"})

        self.assertEqual(
            normalized,
            "SELECT * FROM violation_record l WHERE l.violation_time LIKE '2026-07-24%'",
        )

    def test_does_not_rewrite_non_time_column(self):
        sql = "SELECT * FROM t WHERE date(created_value) = '2026-07-24'"

        self.assertEqual(llm._normalize_exact_date_filters(sql, {"violation_time"}), sql)

    def test_generate_sql_enforces_like_even_if_model_uses_date(self):
        async def fake_chat(_config, messages, temperature=0, max_tokens=None):
            self.assertIn("violation_time\" LIKE 'YYYY-MM-DD%'", messages[-1]["content"])
            model_sql = "SELECT COUNT(*) FROM violation_record l WHERE date(l.violation_time) = '2026-07-24'"
            return llm.ChatResult(
                content=json.dumps({"reasoning_summary": "按违规日期筛选", "sql": model_sql}),
                elapsed_ms=1,
            )

        config = llm.ModelConfig("test", "https://example.invalid/v1", "test")
        columns = [{"name": "violation_time", "type": "TEXT"}]
        route = {"intent_type": "violation_event_stat", "time_fields": ["violation_time"]}

        with patch.object(llm, "chat", fake_chat):
            sql, _, _ = asyncio.run(
                llm.generate_sql(config, "2026年7月24日有多少条违规？", "violation_record", columns, [], route)
            )

        self.assertIn("l.violation_time LIKE '2026-07-24%'", sql)
        self.assertNotIn("date(l.violation_time)", sql)


if __name__ == "__main__":
    unittest.main()
