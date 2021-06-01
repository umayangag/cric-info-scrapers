from team_selection.select_pool import get_player_pool
from itertools import combinations

player_df, wicket_keepers, bowlers = get_player_pool()
# batsmen_only = player_df.loc[player_df['bowling_consistency'] == 0]
file_object = open('combinations.txt', 'a')

player_comb = combinations(player_df["player_name"], 11)
for player_set in player_comb:
    file_object.write(str(list(player_set)) + "\n")
file_object.close()

# for i in range(5, 11):
#     bowlers_comb = combinations(bowlers["player_name"], i)
#     batting_comb = combinations(batsmen_only["player_name"], 11-i)
#
#     for bowler_set in bowlers_comb:
#         for batmen_set in bowlers_comb:
#             team = list(batmen_set) + list(bowler_set)
#             print(team)
#
#             exit()
#     exit()
