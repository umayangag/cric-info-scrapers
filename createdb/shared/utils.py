def int_or_default(value):
    try:
        return int(value)
    except ValueError:
        return "NULL"


def float_or_default(value):
    try:
        return float(value)
    except ValueError:
        return "NULL"

def get_record_id(db_connection, table, value):
    db_cursor = db_connection.cursor()
    db_cursor.execute(f'SELECT id FROM {table} WHERE {table}_name LIKE "{value}"')
    records = db_cursor.fetchall()
    if db_cursor.rowcount == 0:
        db_cursor.execute(f'INSERT INTO {table} SET {table}_name="{value}"')
        db_connection.commit()
        db_cursor.execute(f'SELECT id FROM {table} WHERE {table}_name LIKE "{value}"')
        records = db_cursor.fetchall()
    return records[0][0]