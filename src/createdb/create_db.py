from config.mysql import get_mysql_connection
from createdb.queries import create_db

db_connection = get_mysql_connection()
db_cursor = db_connection.cursor()
db_cursor.execute(create_db.create_db_query)
