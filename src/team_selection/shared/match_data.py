from config.mysql import get_db_connection
from final_data.encoders import *

db_connection = get_db_connection()
db_cursor = db_connection.cursor()


def get_match_data(match_id, label):
    db_cursor.execute(
        f'SELECT inning, {label}_session, toss,venue_id, opposition_id, season_id '
        f'FROM match_details WHERE match_id={match_id} ')
    match_data = db_cursor.fetchall()[0]

    inning = match_data[0]
    session = encode_session(match_data[1])
    toss = match_data[2]
    venue_id = match_data[3]
    opposition_id = match_data[4]
    season_id = match_data[5]

    return inning, session, toss, venue_id, opposition_id, season_id


def get_weather_data(match_id, label):
    db_cursor.execute(
        f'SELECT temp,wind, rain, humidity, cloud, pressure, viscosity '
        f'FROM weather_data WHERE match_id={match_id} and session LIKE "{label}"')

    weather_data = db_cursor.fetchall()[0]

    temp = weather_data[0]
    wind = weather_data[1]
    rain = weather_data[2]
    humidity = weather_data[3]
    cloud = weather_data[4]
    pressure = weather_data[5]
    viscosity = encode_viscosity(weather_data[6])

    return temp, wind, rain, humidity, cloud, pressure, viscosity


def get_player_metric(match_id, label1, player_obj, label2, label3, id):
    search_id = id
    if id == 0:
        search_id = 1
    db_cursor.execute(
        f'SELECT {label1}_{label2} FROM player_{label2}_data WHERE player_id={player_obj[0]} and {label3}_id={search_id}')
    data_list = db_cursor.fetchall()
    if len(data_list) > 0:
        value = data_list[0][0]
        if value > 0:
            return value
    return player_obj[f'{label1}_consistency']


def get_player_list(match_id, label):
    db_cursor.execute(
        f'SELECT player.id, player.player_name, player.is_wicket_keeper, player.is_retired, player.batting_consistency, '
        f'player.bowling_consistency FROM {label}_data '
        f'left join player on player.id={label}_data.player_id '
        f'WHERE match_id = "{match_id}"')
    return db_cursor.fetchall()
