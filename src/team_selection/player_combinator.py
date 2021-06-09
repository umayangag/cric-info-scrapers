from team_selection.create_final_dataset import get_actual_players_who_played
from final_data.encoders import *
import pandas as pd


def actual_team_players(pool_df, match_id):
    actual_player_df, wicket_keepers, bowlers = get_actual_players_who_played(match_id)
    return pool_df[pool_df['player_name'].isin(actual_player_df["player_name"].to_numpy())]


def calculate_overall_performance(input_df, match_id):
    team_df = input_df.copy()
    magic_number = 11 / len(team_df)  # this is to compensate players missing from actual 11
    extras = 14.26

    runs_scored = team_df["runs_scored"].apply(decode_runs)
    balls_faced = team_df["balls_faced"].apply(decode_balls_faced)
    wickets_taken = team_df["wickets_taken"].apply(decode_wickets_taken)
    runs_conceded = team_df["runs_conceded"].apply(decode_runs_conceded)
    deliveries = team_df["deliveries"].apply(decode_deliveries)

    # total_score = runs_scored.sum() * magic_number + extras
    # target = runs_conceded.sum() * magic_number
    # total_balls_faced = balls_faced.sum() * magic_number

    total_score = get_total_score(balls_faced, runs_scored, extras, magic_number)
    target = get_total_conceded(deliveries, runs_conceded, wickets_taken)
    total_balls_faced = calculate_total_balls_faced(balls_faced, magic_number)

    team_df["total_score"] = total_score
    team_df["total_wickets"] = 5
    team_df["total_balls"] = total_balls_faced
    team_df["target"] = target
    team_df["extras"] = extras
    team_df["match_number"] = match_id

    def calculate_batting_contribution(row, key):
        return row[key] / total_score

    def calculate_bowling_contribution(row, key):
        return row[key] / target

    team_df.to_csv("final_team.csv")

    team_df["bowling_contribution"] = team_df.apply(
        lambda row: calculate_bowling_contribution(row, "runs_conceded"), axis=1)
    team_df["batting_contribution"] = team_df.apply(
        lambda row: calculate_batting_contribution(row, "runs_scored"), axis=1)

    print(magic_number, runs_scored.sum(), balls_faced.sum(), extras)
    print(magic_number, runs_conceded.sum(), deliveries.sum(), wickets_taken.sum())
    print("Total Score:", get_total_score(balls_faced, runs_scored, extras, magic_number))
    print("Runs given:", get_total_conceded(deliveries, runs_conceded, wickets_taken))

    # TODO: for evaluation
    # SELECT * FROM `match_details` WHERE `wickets` < 10 AND `balls` < 300 AND `target` IS NOT NULL
    # where the team stopped batting, since they have chased the opposition target before the 50 overs
    # need to compare predicted score with score for 50 overs. because the predicted score will always be high

    return team_df


def get_total_score(balls_faced, runs_scored, extras, magic_number):
    if balls_faced.sum() > 300:
        return (runs_scored.sum() * 300 / balls_faced.sum()) + extras
    if magic_number < 1:
        return (runs_scored.sum() * magic_number) + extras
    return runs_scored.sum() + extras


def get_total_conceded(deliveries, runs_conceded, wickets_taken):
    if deliveries.sum() > 300:
        return runs_conceded.sum() * 300 / deliveries.sum()
    elif wickets_taken.sum() > 10:
        return runs_conceded.sum() * 10 / wickets_taken.sum()
    elif deliveries.sum() < 300:
        return runs_conceded.sum() * 300 / deliveries.sum()

    return runs_conceded.sum()


def calculate_total_balls_faced(balls_faced, magic_number):
    sum = balls_faced.sum() * magic_number
    if sum > 300:
        return 300
    return sum
