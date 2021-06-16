import matplotlib.pyplot as plt

from final_data.batting_regressor import predict_batting
from final_data.bowling_regressor import predict_bowling
from final_data.fielding_regressor import predict_fielding
from final_data.match_win_predict import predict_for_team
from team_selection.dataset_definitions import *
from team_selection.fill_missing_attributes import *
from team_selection.player_combinator import *
from team_selection.shared.match_data import *
import pandas as pd
from final_data.encoders import *

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
    inning2, session2, toss2, venue_id2, opposition_id2, season_id2, score2, total_wickets2, balls2, target2, extras2, match_number2, result2 = get_match_data(
        match_id, "bowling")
    temp, wind, rain, humidity, cloud, pressure, viscosity = get_weather_data(match_id, "batting")
    temp2, wind2, rain2, humidity2, cloud2, pressure2, viscosity2 = get_weather_data(match_id, "bowling")

    data_array = []
    fielding_array = []

    for player in player_list.iterrows():
        player_obj = player[1]
        player_id = player_obj[0]
        player_name = player_obj[1]
        player_form = get_player_metric(match_id, "batting", player_obj, "form", "season", season_id - 1)
        player_venue = get_player_metric(match_id, "batting", player_obj, "venue", "venue", venue_id)
        player_opposition = get_player_metric(match_id, "batting", player_obj, "opposition", "opposition",
                                              opposition_id)
        player_fielding = player_obj[6]

        fielding_array.append([
            player_fielding,
            temp2,
            wind2,
            rain2,
            humidity2,
            cloud2,
            pressure2,
            encode_viscosity(viscosity2),
            inning,
            encode_session(session2),
            toss,
            season_id,
        ])

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
            player_name,
            player_fielding,
        ])
    dataset = pd.DataFrame(data_array, columns=np.concatenate((input_batting_columns, ["fielding_consistency"])))
    fielding_df = pd.DataFrame(fielding_array, columns=input_fielding_columns)
    predicted = predict_batting(dataset.loc[:, dataset.columns != "player_name"])
    success_rate = predict_fielding(fielding_df.loc[:, fielding_df.columns != "player_name"])
    predicted["player_name"] = dataset["player_name"]
    predicted["success_rate"] = success_rate["success_rate"]
    return predicted


def get_player_pool():
    db_cursor.execute(
        f'SELECT id, player_name, is_wicket_keeper,is_retired,batting_consistency,bowling_consistency,fielding_consistency '
        f'FROM player WHERE is_retired=0 and (batting_consistency !=0 or bowling_consistency!=0)')
    player_list = db_cursor.fetchall()
    player_df = pd.DataFrame(player_list, columns=player_columns)
    wicket_keepers = player_df.loc[player_df['is_wicket_keeper'] == 1]
    bowlers = player_df.loc[player_df['bowling_consistency'] > 0]
    return player_df, wicket_keepers, bowlers


def get_player_pool_with_predicted_performance(match_id, players, bowlers_list):
    batting_df = get_batting_performance(players, match_id)
    bowling_df = get_bowling_performance(bowlers_list, match_id)

    bowling_df = bowling_df.loc[:, bowling_df.columns != "toss"]
    bowling_df = bowling_df.loc[:, bowling_df.columns != "season"]
    bowling_df = bowling_df.loc[:, bowling_df.columns != "batting_inning"]
    player_pool = pd.merge(batting_df, bowling_df, on="player_name", how="left")

    calculated_team, score, target = calculate_overall_performance(player_pool, match_id)
    calculated_team = fill_missing_attributes(calculated_team)
    calculated_team_without_names = calculated_team.loc[:, calculated_team.columns != "player_name"]
    # calculated_team_without_names = calculated_team_without_names.loc[:,
    #                                 calculated_team_without_names.columns != "match_number"]
    # calculated_team_without_names = calculated_team_without_names.loc[:,
    #                                 calculated_team_without_names.columns != "season"]
    player_performance_predictions, overall_win_probability = predict_for_team(calculated_team_without_names)
    player_performance_predictions["player_name"] = calculated_team["player_name"]
    return player_performance_predictions


def get_optimal_team_predicted_performance(player_performance_predictions, match_id):
    predicted_team = player_performance_predictions.sort_values(
        by="winning_probability", ascending=False)[:11]

    # COMBINATION ALGORITHM
    predicted_team = player_performance_predictions.copy()
    batsmen_df = predicted_team.loc[predicted_team['bowling_consistency'] == 0]
    batsmen_df = batsmen_df.sort_values(by=["winning_probability", "batting_contribution"], ascending=[False, False])[
                 :6]
    bowler_df = predicted_team.loc[predicted_team['bowling_consistency'] > 0]
    bowler_df = bowler_df.loc[bowler_df['deliveries'] > 30].sort_values(
        by=["winning_probability", "bowling_contribution"], ascending=[False, True])[:5]

    predicted_team = pd.concat([batsmen_df, bowler_df]).drop_duplicates().reset_index(drop=True)
    predicted_team, win_percent = predict_for_team(predicted_team)
    print("WIN % :", win_percent)

    team_df, total_score, target = calculate_overall_performance(predicted_team, match_id)

    # predicted_team2 = player_performance_predictions.copy()
    # batsmen_df2 = predicted_team2.loc[predicted_team2['bowling_consistency'] == 0]
    # batsmen_df2 = batsmen_df2.sort_values(by=["runs_scored", "batting_contribution"], ascending=[False, False])[
    #               :6]
    # bowler_df2 = predicted_team2.loc[predicted_team2['bowling_consistency'] > 0]
    # bowler_df2 = bowler_df2.loc[bowler_df2['deliveries'] > 30].sort_values(
    #     by=["runs_conceded", "bowling_contribution"], ascending=[False, True])[:5]
    #
    # predicted_team2 = pd.concat([batsmen_df2, bowler_df2]).drop_duplicates().reset_index(drop=True)
    # predicted_team2, win_percent2 = predict_for_team(predicted_team2)
    # print("WIN % :", win_percent2)
    #
    # team_df2, total_score2, target2 = calculate_overall_performance(predicted_team2, match_id)
    #
    # if total_score - target > total_score2 - target2:
    #     print("Team A")
    #     return team_df, total_score, target
    # print("Team B")
    # return team_df2, total_score2, target2

    return team_df, total_score, target


def get_actual_team_predicted_performance(player_performance_predictions, match_id):
    actual_team = actual_team_players(player_performance_predictions, match_id)

    return calculate_overall_performance(actual_team, match_id)


# actual wins from test data 14/45==31.11%
if __name__ == "__main__":
    predicted_score_array = []
    # match_id = 1193505
    # match_id = 1198487
    players, keepers, bowlers_list = get_player_pool()

    db_cursor.execute(
        f'SELECT match_details.match_id, score, wickets, balls, total_target FROM match_details '
        f'left join (select match_id, sum(runs) as total_target from bowling_data group by match_id) as runs_conceded '
        f'on match_details.match_id=runs_conceded.match_id WHERE season_id>16 and result !=-1')
    match_list = db_cursor.fetchall()
    matches_df = pd.DataFrame(match_list, columns=["match_id", "score", "wickets", "balls", "target"])
    for match_row in matches_df.iterrows():
        match_id = match_row[1][0]

        # player_pool.to_csv("pool.csv", index=False)
        # player_pool = pd.read_csv("pool.csv")
        player_pool_with_predicted_performance = get_player_pool_with_predicted_performance(match_id, players,
                                                                                            bowlers_list)

        optimal_team, optimal_score, optimal_target = get_optimal_team_predicted_performance(
            player_pool_with_predicted_performance, match_id)

        print("Match_ID", match_id, "Score:", optimal_score, "Target:", optimal_target)
        print(optimal_team.sort_values(by="batting_position", ascending=True)[
                  [
                      "player_name",
                      "runs_scored",
                      "runs_conceded",
                      "economy",
                      "wickets_taken",
                      "winning_probability"
                  ]])

        # ---------------------------------------------------------------

        actual_team, predicted_score, predicted_target = get_actual_team_predicted_performance(
            player_pool_with_predicted_performance, match_id)

        predicted_score_array.append([predicted_score, predicted_target, optimal_score, optimal_target])

        # print("Score:", predicted_score, "Target:", predicted_target)
        # print(actual_team.sort_values(by="batting_position", ascending=True)[
        #           [
        #               "player_name",
        #               "runs_scored",
        #               "runs_conceded",
        #               "economy",
        #               "wickets_taken",
        #               "winning_probability"
        #           ]])
    predicted_totals = pd.DataFrame(predicted_score_array,
                                    columns=["predicted_score", "predicted_target", "optimal_score", "optimal_target"])
    matches_df = pd.concat([matches_df, predicted_totals], axis=1)
    print(matches_df[["score", "balls", "target", "predicted_target"]])

    plt.plot(range(0, len(matches_df["match_id"])), matches_df["score"], color='red', label="actual score")
    plt.plot(range(0, len(matches_df["match_id"])), matches_df["predicted_score"], color='blue',
             label="predicted score")
    plt.plot(range(0, len(matches_df["match_id"])), matches_df["optimal_score"], color='green', label="optimal score")
    plt.title('Actual vs Predicted')
    plt.xlabel('Match Id')
    plt.ylabel('Runs Scored')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(range(0, len(matches_df["match_id"])), matches_df["target"], color='red', label="actual target")
    plt.plot(range(0, len(matches_df["match_id"])), matches_df["predicted_target"], color='blue',
             label="predicted target")
    plt.plot(range(0, len(matches_df["match_id"])), matches_df["optimal_target"], color='green', label="optimal target")
    plt.title('Actual vs Predicted')
    plt.xlabel('Match Id')
    plt.ylabel('Runs Conceded')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(range(0, len(matches_df["match_id"])), matches_df["score"] - matches_df["target"], color='red',
             label="actual margin")
    plt.plot(range(0, len(matches_df["match_id"])), matches_df["predicted_score"] - matches_df["predicted_target"],
             color='blue',
             label="predicted margin")
    plt.plot(range(0, len(matches_df["match_id"])), matches_df["optimal_score"] - matches_df["optimal_target"],
             color='green', label="optimal margin")
    plt.title('Actual vs Predicted')
    plt.xlabel('Match Id')
    plt.ylabel('Winning Margin')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(range(0, len(matches_df["match_id"])), matches_df["optimal_score"], color='red',
             label="score")
    plt.plot(range(0, len(matches_df["match_id"])), matches_df["optimal_target"],
             color='blue',
             label="runs conceded")
    plt.title('Score vs Target')
    plt.xlabel('Match Id')
    plt.ylabel('Runs')
    plt.legend()
    plt.grid()
    plt.show()

    print("Optimal Matches Lost:")
    print(matches_df.loc[matches_df["optimal_score"] < matches_df["optimal_target"]])

    print("Predicted Matches Lost:")
    print(matches_df.loc[matches_df["predicted_score"] < matches_df["predicted_target"]])

    print("Predicted Matches Won:")
    print(matches_df.loc[matches_df["predicted_score"] >= matches_df["predicted_target"]])
