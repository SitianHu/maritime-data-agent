from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile
from openpyxl import Workbook

from app import db
from app.main import import_terms


class TermImportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_PATH", Path(self.temp_dir.name) / "terms.db")
        self.db_patch.start()
        db.init_db()
        with db.connection() as conn:
            conn.execute("ALTER TABLE terms ADD COLUMN rule_key TEXT")

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def workbook_upload(rows: list[list[str]]) -> UploadFile:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["术语", "定义", "同义词", "关联数据表"])
        for row in rows:
            sheet.append(row)
        content = io.BytesIO()
        workbook.save(content)
        content.seek(0)
        return UploadFile(filename="terms.xlsx", file=content)

    def test_import_succeeds_when_terms_table_has_additional_column(self):
        upload = self.workbook_upload([["重点船舶", "需要重点关注的船舶", "重点目标", "全局术语"]])

        result = asyncio.run(import_terms(upload))

        self.assertEqual(result, {"imported": 1, "skipped": 0, "total": 1})
        imported = db.list_terms()
        self.assertEqual(imported[0]["term"], "重点船舶")
        self.assertIsNone(imported[0]["rule_key"])

    def test_single_term_insert_succeeds_when_terms_table_has_additional_column(self):
        item = db.add_term("靠泊", "船舶停靠码头", "停泊", None)

        stored = db.list_terms()[0]
        self.assertEqual(stored["id"], item["id"])
        self.assertIsNone(stored["rule_key"])


if __name__ == "__main__":
    unittest.main()
