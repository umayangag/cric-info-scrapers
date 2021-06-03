from team_selection.create_final_dataset import get_actual_players_who_played


def actual_team_players(pool_df, match_id):
    actual_player_df, wicket_keepers, bowlers = get_actual_players_who_played(match_id)
    return pool_df[pool_df['player_name'].isin(actual_player_df["player_name"].to_numpy())]


def calculate_overall_performance(input_df, match_id):
    team_df = input_df.copy()
    magic_number = 11 / len(team_df)
    extras = 14.26
    total_score = team_df["runs_scored"].sum() * magic_number + extras
    target = team_df["runs_conceded"].sum() * magic_number

    team_df["total_score"] = total_score * magic_number
    team_df["total_wickets"] = 7.59
    team_df["total_balls"] = team_df["balls_faced"].sum() * magic_number
    team_df["target"] = target
    team_df["extras"] = extras
    team_df["match_number"] = match_id

    def calculate_batting_contribution(row, key):
        return row[key] / total_score

    def calculate_bowling_contribution(row, key):
        return row[key] / target

    team_df["bowling_contribution"] = team_df.apply(lambda row: calculate_bowling_contribution(row, "runs_conceded"),
                                                    axis=1)
    team_df["batting_contribution"] = team_df.apply(lambda row: calculate_batting_contribution(row, "runs_scored"),
                                                    axis=1)
    print(total_score, target)
    return team_df
