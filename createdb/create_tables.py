import mysql.connector

create_match_details = "CREATE  TABLE match_details (id INT NOT NULL AUTO_INCREMENT, score INT, wickets INT, overs FLOAT, balls INT, rpo FLOAT, " \
                       "target INT, inning VARCHAR(100), result VARCHAR(20),opposition_id INT, date DATE, match_id INT, " \
                       "batting_session VARCHAR(100), bowling_session VARCHAR(100), venue_id INT, extras INT, toss VARCHAR(10), " \
                       "season_id INT, match_number INT, PRIMARY KEY (id), FOREIGN KEY (venue_id) REFERENCES venue(id), " \
                       "FOREIGN KEY (opposition_id) REFERENCES opposition(id) , FOREIGN KEY (season_id) REFERENCES season(id))"
create_opposition = "CREATE  TABLE opposition (id INT NOT NULL AUTO_INCREMENT, opposition_name VARCHAR(100), PRIMARY KEY (id))"
create_venue = "CREATE  TABLE venue (id INT NOT NULL AUTO_INCREMENT, venue_name VARCHAR(100), PRIMARY KEY (id))"
create_season = "CREATE  TABLE season (id INT NOT NULL AUTO_INCREMENT, season_name VARCHAR(100), PRIMARY KEY (id))"
create_player = "CREATE  TABLE player (id INT NOT NULL AUTO_INCREMENT, player_name VARCHAR(250), PRIMARY KEY (id))"
create_weather_data = "CREATE  TABLE weather_data (id INT NOT NULL AUTO_INCREMENT, match_id INT, session VARCHAR(100), temp INT, feels INT, " \
                      "wind INT, gust INT, rain INT, humidity INT, cloud INT, pressure INT, viscosity VARCHAR(100), " \
                      "PRIMARY KEY (id))"
create_batting_data = "CREATE  TABLE batting_data (id INT NOT NULL AUTO_INCREMENT, match_id INT, player_id INT, " \
                      "description VARCHAR(250), runs INT, balls INT, minutes INT, fours INT, sixes INT, strike_rate float, " \
                      "batting_position INT, PRIMARY KEY (id), FOREIGN KEY (player_id) REFERENCES player(id))"
create_bowling_data = "CREATE  TABLE bowling_data (id INT NOT NULL AUTO_INCREMENT, match_id INT, player_id INT, " \
                      "overs FLOAT, balls INT, maidens INT , runs INT, wickets INT, dots INT, fours INT, sixes INT, econ float, " \
                      "wides INT, no_balls INT, PRIMARY KEY (id), FOREIGN KEY (player_id) REFERENCES player(id))"

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="root",
    auth_plugin='mysql_native_password',
    database="cricket_data"
)
db_cursor = db_connection.cursor()
db_cursor.execute(create_opposition)
db_cursor.execute(create_venue)
db_cursor.execute(create_season)
db_cursor.execute(create_player)
db_cursor.execute(create_match_details)
db_cursor.execute(create_weather_data)
db_cursor.execute(create_batting_data)
db_cursor.execute(create_bowling_data)
