from app.query import validate_sql


def test_allows_select_from_dataset():
    assert validate_sql('SELECT COUNT(*) FROM "sales_123"', 'sales_123').startswith('SELECT')


def test_blocks_write_statement():
    try:
        validate_sql('DELETE FROM "sales_123"', 'sales_123')
        assert False
    except ValueError:
        pass


def test_blocks_other_table():
    try:
        validate_sql('SELECT * FROM users', 'sales_123')
        assert False
    except ValueError:
        pass


def test_allows_cte_over_dataset():
    sql = 'WITH totals AS (SELECT SUM(amount) AS value FROM "sales_123") SELECT * FROM totals'
    assert validate_sql(sql, 'sales_123') == sql
