import mysql.connector

create_match_details = "CREATE  TABLE match_details (id INT NOT NULL AUTO_INCREMENT, score INT, wickets INT, overs FLOAT, balls INT, rpo FLOAT, " \
                       "target INT, inning INT, result VARCHAR(20),opposition_id INT, date VARCHAR(100), match_id INT, " \
                       "batting_session VARCHAR(100), bowling_session VARCHAR(100), venue_id INT, extras INT, toss INT, " \
                       "season_id INT, match_number INT, PRIMARY KEY (id))"
create_opposition = "CREATE  TABLE opposition (id INT NOT NULL AUTO_INCREMENT, opposition_name VARCHAR(100), PRIMARY KEY (id))"
create_venue = "CREATE  TABLE venue (id INT NOT NULL AUTO_INCREMENT, venue_name VARCHAR(100), PRIMARY KEY (id))"
create_season = "CREATE  TABLE season (id INT NOT NULL AUTO_INCREMENT, season_name VARCHAR(100), PRIMARY KEY (id))"

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="root",
    auth_plugin='mysql_native_password',
    database="cricket_data"
)
db_cursor = db_connection.cursor()
db_cursor.execute(create_match_details)
db_cursor.execute(create_opposition)
db_cursor.execute(create_venue)
db_cursor.execute(create_season)
