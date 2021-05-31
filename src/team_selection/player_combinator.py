from team_selection.create_final_dataset import get_actual_players_who_played


def shotlist_players(pool_df, match_id):
    actual_player_df, wicket_keepers, bowlers = get_actual_players_who_played(match_id)
    return pool_df[pool_df['player_name'].isin(actual_player_df["player_name"].to_numpy())]


def calculate_overall_performance(input_df, match_id):
    team_df = input_df.copy()
    if len(team_df) > 11:
        print("invalid number of players in team. only 11 allowed")
        return team_df

    total_score = team_df["runs_scored"].sum()

    def calculate_contribution(row, key):
        return row[key] / total_score

    team_df["total_score"] = total_score
    team_df["total_wickets"] = 7
    team_df["total_balls"] = team_df["balls_faced"].sum()
    team_df["target"] = team_df["runs_scored"].sum()
    team_df["extras"] = 14
    team_df["match_number"] = match_id

    print(team_df)

    team_df["bowling_contribution"] = team_df.apply(lambda row: calculate_contribution(row, "runs_conceded"), axis=1)
    team_df["batting_contribution"] = team_df.apply(lambda row: calculate_contribution(row, "runs_scored"), axis=1)
    return team_df
