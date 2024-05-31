import duckdb


def create_default_connection(filepath=":memory:"):
    db = duckdb.connect(filepath)
    db.install_extension("spatial")
    db.load_extension("spatial")
    db.install_extension("httpfs")
    db.load_extension("httpfs")
    db.execute("SET enable_progress_bar=true;")
    return db


def table_exists(db, table_name):
    query = f"""
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_name = '{table_name}';
    """
    result = db.execute(query).fetchdf()
    return result["count_star()"][0] == 1
