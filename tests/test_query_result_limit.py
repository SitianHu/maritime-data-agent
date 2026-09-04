from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import query


class QueryResultLimitTests(unittest.TestCase):
    def test_readonly_query_returns_more_than_200_rows_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "query.db"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE records (value INTEGER)")
                connection.executemany("INSERT INTO records VALUES (?)", ((value,) for value in range(250)))
                connection.commit()
            finally:
                connection.close()

            with patch.object(query, "DB_PATH", database):
                columns, rows, truncated = query.execute_readonly("SELECT value FROM records ORDER BY value")

        self.assertEqual(columns, ["value"])
        self.assertEqual(len(rows), 250)
        self.assertEqual(rows[-1]["value"], 249)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
