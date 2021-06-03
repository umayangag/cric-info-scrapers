# from itertools import combinations
# from config.mysql import get_db_connection
# import pandas as pd
# from team_selection.dataset_definitions import player_columns
#
# db_connection = get_db_connection()
# db_cursor = db_connection.cursor()
#
#
# def get_player_pool():
#     db_cursor.execute(
#         f'SELECT * FROM player WHERE is_retired=0 and (batting_consistency !=0 or bowling_consistency!=0)')
#     player_list = db_cursor.fetchall()
#     player_df = pd.DataFrame(player_list, columns=player_columns)
#     wicket_keepers = player_df.loc[player_df['is_wicket_keeper'] == 1]
#     bowlers = player_df.loc[player_df['bowling_consistency'] > 0]
#     batsmen = player_df.loc[(player_df['bowling_consistency'] == 0) & (player_df['is_wicket_keeper'] == 0)]
#     return list(batsmen["id"]), list(wicket_keepers["id"]), list(bowlers["id"])
#
#
# batsmen, keepers, bowlers = get_player_pool()
#
# file_object = open('cache/player_combinations.txt', 'a')
#
# for i in range(5, 11):
#     bowler_comb = combinations(bowlers, i)
#     for bowler in bowler_comb:
#         for j in range(1, min(len(keepers), 11 - i) + 1):
#             keeper_comb = combinations(keepers, j)
#             for keeper in keeper_comb:
#                 batsmen_comb = list(combinations(batsmen, 11 - (i + j)))
#                 for batsmen in batsmen_comb:
#                     print(i, j, 11 - (i + j))
#                     file_object.write(
#                         str(list(batsmen) + list(bowlers) + list(keepers)).replace("[", "").replace("]", "") + "\n")
# file_object.close()

# file_object = open('cache/batting_combinations.txt', 'a')


# for i in range(1, 6):
#     print(i)
#     player_comb = combinations(batsmen, i)
#     for player_set in player_comb:
#         print(player_set)
#         file_object.write(str(list(player_set)) + "\n")
# file_object.close()

# file_object = open('bowling_combinations.txt', 'a')
#
# # for i in range(5, 11):
# #     print(i)
# player_comb = combinations(bowlers, 8)
# for player_set in player_comb:
#     print(player_set)
#     file_object.write(str(list(player_set)) + "\n")
# file_object.close()
