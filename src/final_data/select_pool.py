import pandas as pd
from final_data.batting_regressor import predict_batting
from final_data.bowling_regressor import predict_bowling
from team_selection.dataset_definitions import *
from team_selection.shared.match_data import *
from team_selection.player_combinator import *
from final_data.match_win_predict import predict_for_team
from itertools import combinations
from team_selection.fill_missing_attributes import fill_missing_attributes

db_connection = get_db_connection()
db_cursor = db_connection.cursor()


def get_bowling_performance(player_list, match_id):
    inning, session, toss, venue_id, opposition_id, season_id, score, total_wickets, balls, target, extras, match_number, result = get_match_data(
        match_id, "bowling")
    temp, wind, rain, humidity, cloud, pressure, viscosity = get_weather_data(match_id, "bowling")

    data_array = []

    for player in player_list.iterrows():
        player_obj = player[1]
        player_id = player_obj[0]
        player_name = player_obj[1]
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
                           encode_viscosity(viscosity),
                           inning,
                           encode_session(session),
                           toss,
                           player_venue,
                           player_opposition,
                           season_id,
                           player_name])
    dataset = pd.DataFrame(data_array, columns=input_bowling_columns)
    predicted = predict_bowling(dataset.loc[:, dataset.columns != "player_name"])
    predicted["player_name"] = dataset["player_name"]
    return predicted


def get_batting_performance(player_list, match_id):
    inning, session, toss, venue_id, opposition_id, season_id, score, wickets, all_balls, target, extras, match_number, result = get_match_data(
        match_id,
        "batting")
    temp, wind, rain, humidity, cloud, pressure, viscosity = get_weather_data(match_id, "batting")

    data_array = []

    for player in player_list.iterrows():
        player_obj = player[1]
        player_id = player_obj[0]
        player_name = player_obj[1]
        player_form = get_player_metric(match_id, "batting", player_obj, "form", "season", season_id - 1)
        player_venue = get_player_metric(match_id, "batting", player_obj, "venue", "venue", venue_id)
        player_opposition = get_player_metric(match_id, "batting", player_obj, "opposition", "opposition",
                                              opposition_id)

        data_array.append([
            player[1]["batting_consistency"],
            player_form,
            temp,
            wind,
            rain,
            humidity,
            cloud,
            pressure,
            encode_viscosity(viscosity),
            inning,
            encode_session(session),
            toss,
            player_venue,
            player_opposition,
            season_id,
            player_name
        ])
    dataset = pd.DataFrame(data_array, columns=input_batting_columns)
    predicted = predict_batting(dataset.loc[:, dataset.columns != "player_name"])
    predicted["player_name"] = dataset["player_name"]
    return predicted


def get_player_pool():
    db_cursor.execute(
        f'SELECT * FROM player WHERE is_retired=0 and (batting_consistency !=0 or bowling_consistency!=0)')
    player_list = db_cursor.fetchall()
    player_df = pd.DataFrame(player_list, columns=player_columns)
    wicket_keepers = player_df.loc[player_df['is_wicket_keeper'] == 1]
    bowlers = player_df.loc[player_df['bowling_consistency'] > 0]
    return player_df, wicket_keepers, bowlers


if __name__ == "__main__":
    match_id = 1193505
    match_id = 1198487
    players, keepers, bowlers_list = get_player_pool()

    batting_df = get_batting_performance(players, match_id)
    bowling_df = get_bowling_performance(bowlers_list, match_id)

    bowling_df = bowling_df.loc[:, bowling_df.columns != "toss"]
    bowling_df = bowling_df.loc[:, bowling_df.columns != "season"]
    bowling_df = bowling_df.loc[:, bowling_df.columns != "batting_inning"]
    player_pool = pd.merge(batting_df, bowling_df, on="player_name", how="left")
    player_pool = fill_missing_attributes(player_pool)
    player_pool.to_csv("pool.csv", index=False)
    # player_pool = pd.read_csv("pool.csv")

    calculated_team = calculate_overall_performance(player_pool, match_id)
    calculated_team_without_names = calculated_team.loc[:, calculated_team.columns != "player_name"]
    # calculated_team_without_names = calculated_team_without_names.loc[:,
    #                                 calculated_team_without_names.columns != "match_number"]
    # calculated_team_without_names = calculated_team_without_names.loc[:,
    #                                 calculated_team_without_names.columns != "season"]
    player_performance_predictions, overall_win_probability = predict_for_team(calculated_team_without_names)
    player_performance_predictions["player_name"] = calculated_team["player_name"]

    predicted_team = player_performance_predictions.sort_values(
        by="winning_probability", ascending=False)[:11]

    calculated_team = calculate_overall_performance(predicted_team, match_id)

    print(calculated_team.sort_values(by="batting_position", ascending=True)[
              [
                  "player_name",
                  "runs_scored",
                  "runs_conceded",
                  "economy",
                  "wickets_taken",
                  "winning_probability"
              ]])

    # ---------------------------------------------------------------

    actual_team = actual_team_players(player_performance_predictions, match_id)

    actual_team = calculate_overall_performance(actual_team, match_id)

    print(actual_team.sort_values(by="batting_position", ascending=True)[
              [
                  "player_name",
                  "runs_scored",
                  "runs_conceded",
                  "economy",
                  "wickets_taken",
                  "winning_probability"
              ]])
