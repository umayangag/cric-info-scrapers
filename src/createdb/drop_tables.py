from config.mysql import get_db_connection
db_connection = db_connection = get_db_connection()
db_cursor = db_connection.cursor()
db_cursor.execute("drop table player_form_data")
db_cursor.execute("drop table player_venue_data")
db_cursor.execute("drop table player_opposition_data")
db_cursor.execute("drop table batting_data")
db_cursor.execute("drop table bowling_data")
db_cursor.execute("drop table fielding_data")
db_cursor.execute("drop table match_details")
db_cursor.execute("drop table opposition")
db_cursor.execute("drop table venue")
db_cursor.execute("drop table season")
db_cursor.execute("drop table player")
db_cursor.execute("drop table weather_data")

