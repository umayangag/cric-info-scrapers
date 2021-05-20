import mysql.connector

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="root",
    auth_plugin='mysql_native_password',
    database="cricket_data"
)
db_cursor = db_connection.cursor()
db_cursor.execute("drop table batting_data")
db_cursor.execute("drop table bowling_data")
db_cursor.execute("drop table fielding_data")
db_cursor.execute("drop table match_details")
db_cursor.execute("drop table opposition")
db_cursor.execute("drop table venue")
db_cursor.execute("drop table season")
db_cursor.execute("drop table player")
db_cursor.execute("drop table weather_data")

