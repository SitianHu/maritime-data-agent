from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from app import code_dictionary, intent, llm, query
from app.multitable import build_multitable_context, select_terms_for_context


def dataset(dataset_id: str, table_name: str, columns: list[str]) -> dict:
    return {"id": dataset_id, "name": table_name, "table_name": table_name,
            "source_file": f"{table_name}.csv", "columns": [{"name": item, "type": "TEXT"} for item in columns]}


class MultiTablePipelineTests(unittest.TestCase):
    def setUp(self):
        llm.clear_sql_cache()
        self.cross = dataset("cross", "cross_record_line", ["target_id", "event_time", "event_name_code", "event_type_code"])
        self.live = dataset("live", "data_real_time", ["mmsi", "draught", "navstatus"])
        self.status = dataset("status", "vessel_new_status_record", ["target_id", "event_name_code", "event_type_code"])

    def test_second_stage_adds_realtime_and_status_tables(self):
        question = "2026年8月5日进入吴淞VTS区域、当前吃水超过8米且当前处于航行状态的船有多少"
        route = {"dataset_id": "cross", "candidates": [
            {"dataset_id": "cross", "score": 12}, {"dataset_id": "live", "score": 9},
            {"dataset_id": "status", "score": 9}]}
        context = build_multitable_context(question, [self.cross, self.live, self.status], [], route)
        self.assertTrue(context["enabled"])
        self.assertEqual(context["allowed_tables"], ["cross_record_line", "data_real_time", "vessel_new_status_record"])
        self.assertEqual(len(context["join_plan"]), 2)

    def test_current_vessel_fields_resolve_realtime_violation_tie(self):
        violation = dataset("viol", "violation_record", ["target_id", "violation_time"])
        live = dataset("live", "data_real_time", ["mmsi", "ship_name", "draught", "destination"])
        question = "当前区域内，过去7天没有任何违规记录的船舶有哪些？列出船名、当前吃水和目的港。"

        route = intent.route_question(question, [violation, live], [])
        context = build_multitable_context(question, [violation, live], [], route)

        self.assertEqual(route["answer_status"], "success")
        self.assertEqual(route["dataset_id"], "live")
        self.assertEqual(context["allowed_tables"], ["data_real_time", "violation_record"])

    def test_negative_violation_query_uses_current_vessels_as_population(self):
        violation = dataset("viol", "violation_record", ["target_id", "violation_time"])
        live = dataset("live", "data_real_time", ["mmsi", "ship_name"])
        question = "当前区域内，过去7天没有任何违规记录的船舶有哪些？"

        route = intent.route_question(question, [violation, live], [])
        context = build_multitable_context(question, [violation, live], [], route)

        self.assertEqual(route["dataset_id"], "live")
        self.assertEqual(context["primary_table"], "data_real_time")
        self.assertEqual(context["allowed_tables"], ["data_real_time", "violation_record"])

    def test_literal_term_context_does_not_include_unrelated_fallbacks(self):
        terms = [
            {"id": "1", "term": "违规记录", "synonyms": "违规", "definition": "违规事件", "dataset_id": "viol"},
            {"id": "2", "term": "吃水", "synonyms": "draught", "definition": "当前吃水", "dataset_id": "live"},
        ]

        selected = select_terms_for_context(terms, {"viol", "live"}, "过去7天没有违规记录")

        self.assertEqual([item["id"] for item in selected], ["1"])

    def test_number_of_related_tables_has_no_fixed_limit(self):
        extras = [dataset(f"d{i}", f"risk_record_{i}", ["target_id", f"风险{i}"]) for i in range(6)]
        question = " ".join(f"风险{i}" for i in range(6))
        route = {"dataset_id": "cross", "candidates": []}
        context = build_multitable_context(question, [self.cross, *extras], [], route)
        self.assertEqual(len(context["allowed_tables"]), 7)

    def test_sql_prompt_is_controlled_by_multitable_context(self):
        question = "进入区域且吃水超过8米的船有多少"
        context = build_multitable_context(question, [self.cross, self.live], [], {"dataset_id": "cross", "candidates": []})
        captured = {}
        async def fake_chat(_config, messages, temperature=0, max_tokens=None):
            captured["prompt"] = messages[-1]["content"]
            sql = "SELECT COUNT(DISTINCT c.target_id) FROM cross_record_line c JOIN data_real_time l ON c.target_id=l.mmsi"
            return llm.ChatResult(content=json.dumps({"reasoning_summary": "x", "sql": sql}), elapsed_ms=1)
        with patch.object(llm, "chat", fake_chat):
            sql, _, _ = asyncio.run(llm.generate_sql(
                llm.ModelConfig("x", "https://example.invalid/v1", "x"), question,
                self.cross["table_name"], self.cross["columns"], [], {}, {"requests": [], "mappings": []},
                [self.cross, self.live], context))
        self.assertIn("JOIN：", captured["prompt"])
        self.assertIn("只能使用允许访问的 2 张业务表", captured["prompt"])
        query.validate_sql(sql, allowed_tables=context["allowed_tables"])

    def test_negative_event_prompt_requires_not_exists_and_data_relative_window(self):
        violation = dataset("viol", "violation_record", ["target_id", "violation_time"])
        live = dataset("live", "data_real_time", ["mmsi", "ship_name"])
        question = "当前区域内，过去7天没有任何违规记录的船舶有哪些？"
        route = intent.route_question(question, [violation, live], [])
        context = build_multitable_context(question, [violation, live], [], route)
        captured = {}

        async def fake_chat(_config, messages, temperature=0, max_tokens=None):
            captured["prompt"] = messages[-1]["content"]
            sql = "SELECT l.ship_name FROM data_real_time l WHERE NOT EXISTS (SELECT 1 FROM violation_record v WHERE v.target_id=l.mmsi)"
            return llm.ChatResult(content=json.dumps({"reasoning_summary": "x", "sql": sql}), elapsed_ms=1)

        with patch.object(llm, "chat", fake_chat):
            asyncio.run(llm.generate_sql(
                llm.ModelConfig("x", "https://example.invalid/v1", "x"), question,
                live["table_name"], live["columns"], [], route, {"requests": [], "mappings": []},
                [live, violation], context))

        self.assertIn("NOT EXISTS", captured["prompt"])
        self.assertIn("MAX(date", captured["prompt"])

    def test_status_event_suppresses_ais_navstatus_requirement(self):
        with patch.object(code_dictionary, "resolve_code_context", side_effect=[
            {"version": None, "requests": [{"table": "data_real_time", "column": "navstatus", "code_type": "NAV_STATUS"}], "mappings": [], "warnings": []},
            {"version": None, "requests": [], "mappings": [], "warnings": []},
        ]):
            context = code_dictionary.resolve_code_contexts("当前处于航行状态", [self.live, self.status], {})
        self.assertEqual(context["requests"], [])

    def test_answer_disconnect_falls_back_after_successful_query(self):
        trace = {"query_id": "q1", "tables_used": ["a", "b", "c"],
                 "time_range": {"description": "2026-08-05"}, "subject": {"type": "船舶"}}
        with patch.object(llm, "chat", side_effect=RuntimeError("Server disconnected without sending a response")):
            result = asyncio.run(llm.summarize(
                llm.ModelConfig("x", "https://example.invalid/v1", "x"), "有多少船", "SELECT 2",
                ["vessel_count"], [{"vessel_count": 2}], False, {"intent_type": "generic_sql_query"},
                trace, {"direct_answer": "", "scope": {}, "detail": {}, "methodology": {}, "trace_summary": {}}))
        self.assertIn("查询结果为 2", result.content)
        self.assertIn("数据库查询本身已成功", result.content)

    def test_chat_reconnects_with_a_fresh_client_after_gateway_disconnect(self):
        attempts = []

        class FakeClient:
            def __init__(self, **_kwargs):
                self.number = len(attempts)
                attempts.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, url, **_kwargs):
                if self.number < 2:
                    raise httpx.RemoteProtocolError("Server disconnected without sending a response")
                return httpx.Response(
                    200,
                    request=httpx.Request("POST", url),
                    json={"choices": [{"message": {"content": "ok"}}]},
                )

        with patch.object(llm.httpx, "AsyncClient", FakeClient), patch.object(llm.asyncio, "sleep"):
            result = asyncio.run(llm.chat(llm.ModelConfig("x", "https://example.invalid/v1", "x"), []))

        self.assertEqual(result.content, "ok")
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
