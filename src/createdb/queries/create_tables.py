create_match_details = "CREATE TABLE match_details (id INT NOT NULL AUTO_INCREMENT, score INT, wickets INT, overs FLOAT, balls INT, rpo FLOAT, " \
                       "target INT, inning INT, result INT,opposition_id INT, date DATE, match_id INT, " \
                       "batting_session VARCHAR(100), bowling_session VARCHAR(100), venue_id INT, extras INT, toss VARCHAR(10), " \
                       "season_id INT, match_number INT, PRIMARY KEY (id), FOREIGN KEY (venue_id) REFERENCES venue(id), " \
                       "FOREIGN KEY (opposition_id) REFERENCES opposition(id) , FOREIGN KEY (season_id) REFERENCES season(id))"

create_opposition = "CREATE TABLE opposition (id INT NOT NULL AUTO_INCREMENT, opposition_name VARCHAR(100), PRIMARY KEY (id))"

create_venue = "CREATE TABLE venue (id INT NOT NULL AUTO_INCREMENT, venue_name VARCHAR(100), PRIMARY KEY (id))"

create_season = "CREATE TABLE season (id INT NOT NULL AUTO_INCREMENT, season_name VARCHAR(100), PRIMARY KEY (id))"

create_player = "CREATE TABLE player (id INT NOT NULL AUTO_INCREMENT, player_name VARCHAR(250), is_wicket_keeper INT, " \
                "is_retired INT, batting_consistency FLOAT, bowling_consistency FLOAT, fielding_consistency FLOAT, PRIMARY KEY (id))"

create_weather_data = "CREATE TABLE weather_data (id INT NOT NULL AUTO_INCREMENT, match_id INT, session VARCHAR(100), temp INT, feels INT, " \
                      "wind INT, gust INT, rain INT, humidity INT, cloud INT, pressure INT, viscosity VARCHAR(100), " \
                      "PRIMARY KEY (id))"

create_batting_data = "CREATE TABLE batting_data (id INT NOT NULL AUTO_INCREMENT, match_id INT, player_id INT, " \
                      "description VARCHAR(250), runs INT, balls INT, minutes INT, fours INT, sixes INT, strike_rate float, " \
                      "batting_position INT, PRIMARY KEY (id), FOREIGN KEY (player_id) REFERENCES player(id))"

create_bowling_data = "CREATE TABLE bowling_data (id INT NOT NULL AUTO_INCREMENT, match_id INT, player_id INT, " \
                      "overs FLOAT, balls INT, maidens INT , runs INT, wickets INT, dots INT, fours INT, sixes INT, econ float, " \
                      "wides INT, no_balls INT, PRIMARY KEY (id), FOREIGN KEY (player_id) REFERENCES player(id))"

create_fielding_data = "CREATE TABLE fielding_data (id INT NOT NULL AUTO_INCREMENT, match_id INT, player_id INT, " \
                       "catches INT, run_outs INT, dropped_catches INT, missed_run_outs INT, PRIMARY KEY (id), FOREIGN KEY (player_id) REFERENCES player(id))"

create_player_form_data = "CREATE TABLE player_form_data (id INT NOT NULL AUTO_INCREMENT, player_id INT, " \
                          "season_id INT, batting_form FLOAT, bowling_form FLOAT, PRIMARY KEY (id), FOREIGN KEY (player_id) REFERENCES player(id)" \
                          ", FOREIGN KEY (season_id) REFERENCES season(id))"

create_player_venue_data = "CREATE TABLE player_venue_data (id INT NOT NULL AUTO_INCREMENT, player_id INT, " \
                           "venue_id INT, batting_venue FLOAT, bowling_venue FLOAT, PRIMARY KEY (id), FOREIGN KEY (player_id) REFERENCES player(id)" \
                           ", FOREIGN KEY (venue_id) REFERENCES venue(id))"

create_player_opposition_data = "CREATE TABLE player_opposition_data (id INT NOT NULL AUTO_INCREMENT, player_id INT, " \
                          "opposition_id INT, batting_opposition FLOAT, bowling_opposition FLOAT, PRIMARY KEY (id), " \
                                "FOREIGN KEY (player_id) REFERENCES player(id)" \
                          ", FOREIGN KEY (opposition_id) REFERENCES opposition(id))"