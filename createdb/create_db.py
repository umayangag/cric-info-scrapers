import mysql.connector

create_db = "CREATE DATABASE cricket_data"

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="root",
    auth_plugin='mysql_native_password',
)
db_cursor = db_connection.cursor()
db_cursor.execute(create_db)