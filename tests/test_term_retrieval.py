from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import db, term_retrieval


class TermRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_PATH", Path(self.temp_dir.name) / "terms.db")
        self.db_patch.start()
        db.init_db()
        with db.connection() as conn:
            for dataset_id in ("one", "two"):
                conn.execute(
                    "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (dataset_id, dataset_id, f"table_{dataset_id}", "test.csv", 0, "[]", db.utc_now()),
                )

    def tearDown(self):
        term_retrieval._model.cache_clear()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_keyword_scope_and_dataset_override(self):
        db.add_term("吃水", "全局定义", "水下深度", None)
        local = db.add_term("吃水", "数据集定义", "船舶吃水", "one")
        db.add_term("专属术语", "仅二号可用", "秘密规则", "two")
        with patch.object(term_retrieval, "SEMANTIC_ENABLED", False):
            hits, trace = term_retrieval.retrieve_terms("吃水超过8米的船有哪些？", "one")
            isolated, _ = term_retrieval.retrieve_terms("秘密规则是什么？", "one")
        self.assertEqual([item["id"] for item in hits], [local["id"]])
        self.assertEqual(hits[0]["match_sources"], ["keyword"])
        self.assertEqual(trace["method"], "keyword")
        self.assertEqual(isolated, [])

    def test_semantic_merge_irrelevant_and_fallback(self):
        db.add_term("危险船舶", "风险评分超过80的船舶", "高风险目标", "one")
        db.add_term("靠泊状态", "船舶已停靠码头", "在泊", "one")

        def fake_embed(texts):
            vectors = []
            for text in texts:
                if "重点关注" in text or "危险船舶" in text:
                    vectors.append([1.0, 0.0, 0.0])
                elif "靠泊状态" in text:
                    vectors.append([0.0, 1.0, 0.0])
                else:
                    vectors.append([0.0, 0.0, 1.0])
            return vectors

        with patch.object(term_retrieval, "_embed", fake_embed):
            hits, trace = term_retrieval.retrieve_terms("哪些目标需要重点关注？", "one")
            irrelevant, _ = term_retrieval.retrieve_terms("今天天气怎么样？", "one")
        self.assertEqual([item["term"] for item in hits], ["危险船舶"])
        self.assertEqual(hits[0]["match_sources"], ["semantic"])
        self.assertEqual(trace["method"], "hybrid")
        self.assertEqual(irrelevant, [])

        with patch.object(term_retrieval, "_embed", side_effect=RuntimeError("offline")):
            fallback, trace = term_retrieval.retrieve_terms("高风险目标有哪些？", "one")
        self.assertEqual([item["term"] for item in fallback], ["危险船舶"])
        self.assertEqual(trace["method"], "keyword_fallback")


if __name__ == "__main__":
    unittest.main()
