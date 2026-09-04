import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import db


class DatasetRenameTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_PATH", Path(self.temp_dir.name) / "datasets.db")
        self.db_patch.start()
        db.init_db()
        db.save_dataset("one", "原名称", "table_one", "one.csv", 0, [])
        db.save_dataset("two", "另一张表", "table_two", "two.csv", 0, [])

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_rename_changes_only_display_name(self):
        renamed = db.rename_dataset("one", "  新术语表名称  ")

        self.assertEqual(renamed["name"], "新术语表名称")
        self.assertEqual(renamed["table_name"], "table_one")
        self.assertEqual(db.get_dataset("one")["source_file"], "one.csv")

    def test_rename_rejects_blank_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "不能为空"):
            db.rename_dataset("one", "   ")
        with self.assertRaisesRegex(ValueError, "已存在"):
            db.rename_dataset("one", "另一张表")

    def test_rename_returns_none_for_unknown_dataset(self):
        self.assertIsNone(db.rename_dataset("missing", "新名称"))


if __name__ == "__main__":
    unittest.main()
