from config.mysql import get_db_connection
import pandas as pd
from final_data.queries import batting_win_dataset_query
import os
from analyze.cluster_batting import cluster_batting_performance
from analyze.normalize_dataset import normalize_dataset
from final_data.encoders import *
from sklearn import preprocessing

dirname = os.path.dirname(__file__)
input_file_encoded = os.path.join(dirname, "output\\match_weather.csv")
output_file_encoded = os.path.join(dirname, "output\\overall.csv")
db_connection = get_db_connection()
db_cursor = db_connection.cursor()
df = pd.read_csv(input_file_encoded)
player_columns = ["player_id",
                  "description",
                  "runs_scored",
                  "balls_faced",
                  "minutes",
                  "fours",
                  "sixes",
                  "strike_rate",
                  "batting_position",
                  "overs",
                  "deliveries",
                  "maidens",
                  "runs_conceeded",
                  "wickets_taken",
                  "dots",
                  "fours",
                  "sixes",
                  "econ",
                  "wides",
                  "no_balls"]

for column in df.columns:
    player_columns.append(column)
player_df = pd.DataFrame(columns=player_columns)

for i, row in df.iterrows():
    match_id = row["match_id"]
    player_list = {}
    db_cursor.execute(
        f'select player_id, description, runs, balls, minutes, fours, sixes, strike_rate, batting_position from batting_data where match_id="{match_id}"')
    batsmen_ids = db_cursor.fetchall()
    db_cursor.execute(
        f'select player_id, overs, balls, maidens,runs, wickets, dots, fours, sixes, econ, wides, no_balls from bowling_data where match_id="{match_id}"')
    bowler_ids = db_cursor.fetchall()

    for j in batsmen_ids:
        player_list[j[0]] = {}
        player_list[j[0]]["batting"] = j
        player_list[j[0]]["bowling"] = {}
    for k in bowler_ids:
        if k[0] not in player_list.keys():
            player_list[k[0]] = {}
            player_list[k[0]]["batting"] = {}
            player_list[k[0]]["bowling"] = {}
        player_list[k[0]]["bowling"] = k

    for index in player_list:
        row["player_id"] = index
        if len(player_list[index]["batting"]) == 0:
            row["description"] = 0
            row["runs_scored"] = 0
            row["balls_faced"] = 0
            row["minutes"] = 0
            row["fours"] = 0
            row["sixes"] = 0
            row["strike_rate"] = 0
            row["batting_position"] = 0
        else:
            row["description"] = player_list[index]["batting"][1]
            row["runs_scored"] = player_list[index]["batting"][2]
            row["balls_faced"] = player_list[index]["batting"][3]
            row["minutes"] = player_list[index]["batting"][4]
            row["fours"] = player_list[index]["batting"][5]
            row["sixes"] = player_list[index]["batting"][6]
            row["strike_rate"] = player_list[index]["batting"][7]
            row["batting_position"] = player_list[index]["batting"][8]
        if len(player_list[index]["bowling"]) == 0:
            row["overs"] = 0
            row["deliveries"] = 0
            row["maidens"] = 0
            row["runs_conceeded"] = 0
            row["wickets_taken"] = 0
            row["dots"] = 0
            row["fours"] = 0
            row["sixes"] = 0
            row["econ"] = 0
            row["wides"] = 0
            row["no_balls"] = 0
        else:
            row["overs"] = player_list[index]["bowling"][1]
            row["deliveries"] =player_list[index]["bowling"][2]
            row["maidens"] = player_list[index]["bowling"][3]
            row["runs_conceeded"] = player_list[index]["bowling"][4]
            row["wickets_taken"] = player_list[index]["bowling"][5]
            row["dots"] = player_list[index]["bowling"][6]
            row["fours"] = player_list[index]["bowling"][7]
            row["sixes"] = player_list[index]["bowling"][8]
            row["econ"] =player_list[index]["bowling"][9]
            row["wides"] =player_list[index]["bowling"][10]
            row["no_balls"] = player_list[index]["bowling"][11]
        player_df = player_df.append(row)

    break

if os.path.exists(output_file_encoded):
    print("existing file deleted")
    os.remove(output_file_encoded)
player_df.to_csv(output_file_encoded, index=False)
