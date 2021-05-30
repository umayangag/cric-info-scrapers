import pandas as pd
import os
from final_data.bowling_regressor import predict_bowling
from team_selection.shared.match_data import *
from team_selection.dataset_definitions import *

dirname = os.path.dirname(__file__)
output_file_encoded = os.path.join(dirname, "output\\final_dataset.csv")
db_connection = get_db_connection()
db_cursor = db_connection.cursor()


def get_bowling_df(player_list, match_id):
    inning, session, toss, venue_id, opposition_id, season_id = get_match_data(match_id, "bowling")
    temp, wind, rain, humidity, cloud, pressure, viscosity = get_weather_data(match_id, "bowling")

    data_array = []

    for player in player_list.iterrows():
        player_obj = player[1]
        player_id = player_obj[0]
        player_form = get_player_metric(match_id, "bowling", player_obj, "form", "season", season_id - 1)
        player_venue = get_player_metric(match_id, "bowling", player_obj, "venue", "venue", venue_id)
        player_opposition = get_player_metric(match_id, "bowling", player_obj, "opposition", "opposition",
                                              opposition_id)
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
                           session,
                           toss,
                           player_venue,
                           player_opposition,
                           season_id,
                           player_id])
    return pd.DataFrame(data_array, columns=input_bowling_columns)


def get_batting_df(player_list, match_id):
    inning, session, toss, venue_id, opposition_id, season_id = get_match_data(match_id, "batting")
    temp, wind, rain, humidity, cloud, pressure, viscosity = get_weather_data(match_id, "batting")

    data_array = []

    for player in player_list.iterrows():
        player_obj = player[1]
        player_id = player_obj[0]
        player_form = get_player_metric(match_id, "batting", player_obj, "form", "season", season_id - 1)
        player_venue = get_player_metric(match_id, "batting", player_obj, "venue", "venue", venue_id)
        player_opposition = get_player_metric(match_id, "batting", player_obj, "opposition", "opposition",
                                              opposition_id)

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
                           session,
                           toss,
                           player_venue,
                           player_opposition,
                           season_id,
                           player_id])
    return pd.DataFrame(data_array, columns=input_batting_columns)


def get_players(match_id):
    batsmen = get_player_list(match_id, "batting")
    bowlers = get_player_list(match_id, "bowling")
    player_list = batsmen + list(set(bowlers) - set(batsmen))

    player_df = pd.DataFrame(player_list, columns=player_columns)
    wicket_keepers = player_df.loc[player_df['is_wicket_keeper'] == 1]
    bowlers = player_df.loc[player_df['bowling_consistency'] > 0]
    return player_df, wicket_keepers, bowlers


match_id = 1193505
players, keepers, bowlers_list = get_players(match_id)
batting_df = get_batting_df(players, match_id)
bowling_df = get_bowling_df(bowlers_list, match_id)
# bowling_df = bowling_df.loc[:, bowling_df.columns != "toss"]
# bowling_df = bowling_df.loc[:, bowling_df.columns != "season"]
# bowling_df = bowling_df.loc[:, bowling_df.columns != "batting_inning"]
final_df = pd.merge(batting_df, bowling_df, on="player_name", how="left").fillna(0)
final_df.to_csv("final_dataset.csv")
