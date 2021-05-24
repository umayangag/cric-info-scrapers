from config import get_db_connection
from queries import create_tables
db_connection = get_db_connection()
db_cursor = db_connection.cursor()
db_cursor.execute(create_tables.create_opposition)
db_cursor.execute(create_tables.create_venue)
db_cursor.execute(create_tables.create_season)
db_cursor.execute(create_tables.create_player)
db_cursor.execute(create_tables.create_match_details)
db_cursor.execute(create_tables.create_weather_data)
db_cursor.execute(create_tables.create_batting_data)
db_cursor.execute(create_tables.create_bowling_data)
db_cursor.execute(create_tables.create_fielding_data)
