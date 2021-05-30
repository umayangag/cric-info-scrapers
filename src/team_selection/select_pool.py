from config.mysql import get_db_connection
import pandas as pd
import itertools
from final_data.queries import batting_dataset_query
import os
import numpy as np
from analyze.cluster_batting import cluster_batting_performance
from analyze.normalize_batting import normalize_batting_dataset
from final_data.encoders import *
from final_data.batting_regressor import predict_batting
from final_data.bowling_regressor import predict_bowling

columns = [
    "id",
    "player_name",
    "is_wicket_keeper",
    "is_retired",
    "batting_consistency",
    "bowling_consistency",
]

predict_batting_columns = [
    # "batting_position",
    "batting_consistency",
    "batting_form",
    "batting_temp",
    "batting_wind",
    "batting_rain",
    "batting_humidity",
    "batting_cloud",
    "batting_pressure",
    "batting_viscosity",
    "batting_inning",
    "batting_session",
    "toss",
    "venue",
    "opposition",
    "season",
]

predict_bowling_columns = [
    "bowling_consistency",
    "bowling_form",
    "bowling_temp",
    "bowling_wind",
    "bowling_rain",
    "bowling_humidity",
    "bowling_cloud",
    "bowling_pressure",
    "bowling_viscosity",
    "bowling_inning",
    "bowling_session",
    "toss",
    "venue",
    "opposition",
    "season",
]

final_columns = ['player_id', 'runs_scored', 'balls_faced', 'fours_scored',
                 'sixes_scored', 'strike_rate', 'batting_position', 'overs_bowled',
                 'deliveries', 'maidens', 'runs_conceded', 'wickets_taken', 'dots',
                 'fours_given', 'sixes_given', 'econ', 'wides', 'no_balls', 'score',
                 'wickets', 'overs', 'balls', 'inning', 'result', 'opposition_id',
                 'venue_id', 'toss', 'season_id', 'match_number', 'batting_temp',
                 'batting_feels', 'batting_wind', 'batting_gust', 'batting_rain',
                 'batting_humidity', 'batting_cloud', 'batting_pressure', 'bowling_temp',
                 'bowling_feels', 'bowling_wind', 'bowling_gust', 'bowling_rain',
                 'bowling_humidity', 'bowling_cloud', 'bowling_pressure',
                 'batting_contribution', 'bowling_contribution']

dirname = os.path.dirname(__file__)
output_file_encoded = os.path.join(dirname, "output\\player_pool.csv")
db_connection = get_db_connection()
db_cursor = db_connection.cursor()


def get_bowling_performance(player_list, match_id):
    db_cursor.execute(
        f'SELECT inning, bowling_session, toss,venue_id, opposition_id, season_id FROM match_details WHERE match_id={match_id} ')
    match_data = db_cursor.fetchall()[0]

    inning = match_data[0]
    bowling_session = encode_session(match_data[1])
    toss = match_data[2]
    venue_id = match_data[3]
    opposition_id = match_data[4]
    season_id = match_data[5]

    db_cursor.execute(
        f'SELECT temp,wind, rain, humidity, cloud, pressure, viscosity FROM weather_data WHERE match_id={match_id} and session LIKE "bowling"')

    weather_data = db_cursor.fetchall()[0]

    temp = weather_data[0]
    wind = weather_data[1]
    rain = weather_data[2]
    humidity = weather_data[3]
    cloud = weather_data[4]
    pressure = weather_data[5]
    viscosity = encode_viscosity(weather_data[6])

    data_array = []

    for player in player_list.iterrows():
        db_cursor.execute(
            f'SELECT bowling_form FROM player_form_data WHERE player_id={player[0]} and season_id={season_id-1}')
        form_data = db_cursor.fetchall()
        if len(form_data) > 0:
            player_form = form_data[0][0]
        else:
            player_form = player[1]["bowling_consistency"]

        db_cursor.execute(
            f'SELECT bowling_venue FROM player_venue_data WHERE player_id={player[0]} and venue_id={venue_id}')
        venue_data = db_cursor.fetchall()
        if len(venue_data) > 0:
            venue = venue_data[0][0]
        else:
            venue = player[1]["bowling_consistency"]

        db_cursor.execute(
            f'SELECT bowling_opposition FROM player_opposition_data WHERE player_id={player[0]} and opposition_id={opposition_id}')
        opposition_data = db_cursor.fetchall()
        if len(opposition_data) > 0:
            opposition = opposition_data[0][0]
        else:
            opposition = player[1]["bowling_consistency"]
        data_array.append([player[1]["bowling_consistency"],
                           player_form,
                           temp,
                           wind,
                           rain,
                           humidity,
                           cloud,
                           pressure,
                           viscosity,
                           inning,
                           bowling_session,
                           toss,
                           venue,
                           opposition,
                           season_id])
    dataset = pd.DataFrame(data_array, columns=predict_bowling_columns)
    return predict_bowling(dataset)


def get_batting_performance(player_list, match_id):
    db_cursor.execute(
        f'SELECT inning, batting_session, toss,venue_id, opposition_id, season_id FROM match_details WHERE match_id={match_id} ')
    match_data = db_cursor.fetchall()[0]

    inning = match_data[0]
    batting_session = encode_session(match_data[1])
    toss = match_data[2]
    venue_id = match_data[3]
    opposition_id = match_data[4]
    season_id = match_data[5]

    db_cursor.execute(
        f'SELECT temp,wind, rain, humidity, cloud, pressure, viscosity FROM weather_data WHERE match_id={match_id} and session LIKE "batting"')

    weather_data = db_cursor.fetchall()[0]

    temp = weather_data[0]
    wind = weather_data[1]
    rain = weather_data[2]
    humidity = weather_data[3]
    cloud = weather_data[4]
    pressure = weather_data[5]
    viscosity = encode_viscosity(weather_data[6])

    data_array = []

    for player in player_list.iterrows():
        db_cursor.execute(
            f'SELECT batting_form FROM player_form_data WHERE player_id={player[0]} and season_id={season_id-1}')
        form_data = db_cursor.fetchall()
        if len(form_data) > 0:
            player_form = form_data[0][0]
        else:
            player_form = player[1]["batting_consistency"]

        db_cursor.execute(
            f'SELECT batting_venue FROM player_venue_data WHERE player_id={player[0]} and venue_id={venue_id}')
        venue_data = db_cursor.fetchall()
        if len(venue_data) > 0:
            venue = venue_data[0][0]
        else:
            venue = player[1]["batting_consistency"]

        db_cursor.execute(
            f'SELECT batting_opposition FROM player_opposition_data WHERE player_id={player[0]} and opposition_id={opposition_id}')
        opposition_data = db_cursor.fetchall()
        if len(opposition_data) > 0:
            opposition = opposition_data[0][0]
        else:
            opposition = player[1]["batting_consistency"]
        data_array.append([player[1]["batting_consistency"],
                           player_form,
                           temp,
                           wind,
                           rain,
                           humidity,
                           cloud,
                           pressure,
                           viscosity,
                           inning,
                           batting_session,
                           toss,
                           venue,
                           opposition,
                           season_id])
    dataset = pd.DataFrame(data_array, columns=predict_batting_columns)
    return predict_batting(dataset)


def get_player_pool():
    db_cursor.execute(
        f'SELECT * FROM player WHERE is_retired=0 and (batting_consistency !=0 or bowling_consistency!=0)')
    player_list = db_cursor.fetchall()
    player_df = pd.DataFrame(player_list, columns=columns)
    wicket_keepers = player_df.loc[player_df['is_wicket_keeper'] == 1]
    bowlers = player_df.loc[player_df['bowling_consistency'] > 0]
    return player_df, wicket_keepers, bowlers


players, keepers, bowlers_list = get_player_pool()
match_id = 1193505
batting_df = get_batting_performance(players, match_id)
bowling_df = get_bowling_performance(players, match_id)

# final_df=batting_df.app
# batting_df.to_csv("batting_test.csv", index=False)
# bowling_df.to_csv("bowling_test.csv", index=False)
print(batting_df.columns)
print(bowling_df.columns)
