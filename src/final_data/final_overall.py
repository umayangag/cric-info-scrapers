import os

import pandas as pd

from config.mysql import get_db_connection

dirname = os.path.dirname(__file__)
input_file_encoded = os.path.join(dirname, "output\\match_weather.csv")
output_file_encoded = os.path.join(dirname, "output\\overall.csv")
db_connection = get_db_connection()
db_cursor = db_connection.cursor()
df = pd.read_csv(input_file_encoded)
player_columns = ['batting_consistency', 'batting_form', 'batting_temp', 'batting_wind',
                  'batting_rain', 'batting_humidity', 'batting_cloud', 'batting_pressure',
                  'batting_viscosity', 'batting_inning', 'batting_session', 'toss',
                  'batting_venue', 'batting_opposition', 'season', 'runs_scored', 'balls_faced',
                  'fours_scored', 'sixes_scored', 'batting_position', 'strike_rate', 'bowling_consistency',
                  'bowling_form', 'bowling_temp',
                  'bowling_wind', 'bowling_rain', 'bowling_humidity', 'bowling_cloud',
                  'bowling_pressure', 'bowling_viscosity',
                  'bowling_session', 'bowling_venue', 'bowling_opposition', 'runs_conceded', 'deliveries',
                  'wickets_taken',
                  'econ']

for column in df.columns:
    player_columns.append(column)
player_df = pd.DataFrame(columns=player_columns)
data_array = []


def calculate_batting_contribution(rowitem):
    if rowitem["runs_scored"] == 0 or rowitem["score"] == 0:
        return 0
    return rowitem["runs_scored"] / rowitem["score"]


def calculate_bowling_contribution(rowitem):
    if rowitem["runs_conceded"] == 0 or rowitem["score"] == 0:
        return 0
    return rowitem["runs_conceded"] / rowitem["score"]


if os.path.exists(output_file_encoded):
    print("existing file deleted")
    os.remove(output_file_encoded)

for i, row in df.iterrows():
    if row["match_number"] not in [4052, 3092, 3274, 3580, 3718, 3807]:
        match_id = row["match_id"]
        player_list = {}
        db_cursor.execute(
            f'select player.batting_consistency, player_form_data.batting_form, weather_data.temp, weather_data.wind, '
            f'weather_data.rain, weather_data.humidity, weather_data.cloud, weather_data.pressure, weather_data.viscosity, '
            f'match_details.inning, match_details.batting_session, match_details.toss, player_venue_data.batting_venue, '
            f'player_opposition_data.batting_opposition, match_details.season_id, runs, batting_data.balls, fours, sixes, '
            f'batting_data.batting_position, strike_rate, '
            f'batting_position from batting_data left join player on player.id=batting_data.player_id '
            f'left join match_details on match_details.match_id=batting_data.match_id '
            f'left join player_venue_data on player_venue_data.player_id=batting_data.player_id and player_venue_data.venue_id=match_details.venue_id '
            f'left join player_opposition_data on player_opposition_data.player_id=batting_data.player_id and player_opposition_data.opposition_id=match_details.opposition_id '
            f'left join weather_data on batting_data.match_id=weather_data.match_id and weather_data.session="batting"'
            f'left join player_form_data on player_form_data.player_id=batting_data.player_id and player_form_data.season_id=match_details.season_id'
            f' where batting_data.match_id="{match_id}"')
        batsmen_ids = db_cursor.fetchall()
        db_cursor.execute(
            f'select player.bowling_consistency, player_form_data.bowling_form, weather_data.temp, weather_data.wind, '
            f'weather_data.rain, weather_data.humidity, weather_data.cloud, weather_data.pressure, weather_data.viscosity, '
            f'match_details.bowling_session, player_venue_data.bowling_venue, '
            f'player_opposition_data.bowling_opposition, runs, bowling_data.balls, bowling_data.wickets, econ '
            f'from bowling_data left join player on player.id=bowling_data.player_id '
            f'left join match_details on match_details.match_id=bowling_data.match_id '
            f'left join player_venue_data on player_venue_data.player_id=bowling_data.player_id and player_venue_data.venue_id=match_details.venue_id '
            f'left join player_opposition_data on player_opposition_data.player_id=bowling_data.player_id and player_opposition_data.opposition_id=match_details.opposition_id '
            f'left join weather_data on bowling_data.match_id=weather_data.match_id and weather_data.session="bowling"'
            f'left join player_form_data on player_form_data.player_id=bowling_data.player_id and player_form_data.season_id=match_details.season_id'
            f' where bowling_data.match_id="{match_id}"')
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
                row['batting_consistency'] = 0
                row['batting_form'] = 0
                row['batting_temp'] = 0
                row['batting_wind'] = 0
                row['batting_rain'] = 0
                row['batting_humidity'] = 0
                row['batting_cloud'] = 0
                row['batting_pressure'] = 0
                row['batting_viscosity'] = 0
                row['batting_inning'] = 0
                row['batting_session'] = 0
                row['toss'] = 0
                row['batting_venue'] = 0
                row['batting_opposition'] = 0
                row['season'] = 0
                row['runs_scored'] = 0
                row['balls_faced'] = 0
                row['fours_scored'] = 0
                row['sixes_scored'] = 0
                row['batting_position'] = 0
                row['strike_rate'] = 0
            else:
                row['batting_consistency'] = player_list[index]["batting"][0]
                row['batting_form'] = player_list[index]["batting"][1]
                row['batting_temp'] = player_list[index]["batting"][2]
                row['batting_wind'] = player_list[index]["batting"][3]
                row['batting_rain'] = player_list[index]["batting"][4]
                row['batting_humidity'] = player_list[index]["batting"][5]
                row['batting_cloud'] = player_list[index]["batting"][6]
                row['batting_pressure'] = player_list[index]["batting"][7]
                row['batting_viscosity'] = player_list[index]["batting"][8]
                row['batting_inning'] = player_list[index]["batting"][9]
                row['batting_session'] = player_list[index]["batting"][10]
                row['toss'] = player_list[index]["batting"][11]
                row['batting_venue'] = player_list[index]["batting"][12]
                row['batting_opposition'] = player_list[index]["batting"][13]
                row['season'] = player_list[index]["batting"][14]
                row['runs_scored'] = player_list[index]["batting"][15]
                row['balls_faced'] = player_list[index]["batting"][16]
                row['fours_scored'] = player_list[index]["batting"][17]
                row['sixes_scored'] = player_list[index]["batting"][18]
                row['batting_position'] = player_list[index]["batting"][19]
                row['strike_rate'] = player_list[index]["batting"][20]
            if len(player_list[index]["bowling"]) == 0:
                row['bowling_consistency'] = 0
                row['bowling_form'] = 0
                row['bowling_temp'] = 0
                row['bowling_wind'] = 0
                row['bowling_rain'] = 0
                row['bowling_humidity'] = 0
                row['bowling_cloud'] = 0
                row['bowling_pressure'] = 0
                row['bowling_viscosity'] = 0
                row['bowling_session'] = 0
                row['bowling_venue'] = 0
                row['bowling_opposition'] = 0
                row['runs_conceded'] = 0
                row['deliveries'] = 0
                row['wickets_taken'] = 0
                row['econ'] = 0
            else:
                row['bowling_consistency'] = player_list[index]["bowling"][0]
                row['bowling_form'] = player_list[index]["bowling"][1]
                row['bowling_temp'] = player_list[index]["bowling"][2]
                row['bowling_wind'] = player_list[index]["bowling"][3]
                row['bowling_rain'] = player_list[index]["bowling"][4]
                row['bowling_humidity'] = player_list[index]["bowling"][5]
                row['bowling_cloud'] = player_list[index]["bowling"][6]
                row['bowling_pressure'] = player_list[index]["bowling"][7]
                row['bowling_viscosity'] = player_list[index]["bowling"][8]
                row['bowling_session'] = player_list[index]["bowling"][9]
                row['bowling_venue'] = player_list[index]["bowling"][10]
                row['bowling_opposition'] = player_list[index]["bowling"][11]
                row['runs_conceded'] = player_list[index]["bowling"][12]
                row['deliveries'] = player_list[index]["bowling"][13]
                row['wickets_taken'] = player_list[index]["bowling"][14]
                row['econ'] = player_list[index]["bowling"][15]

            data_array.append(row)

player_df = pd.DataFrame(data_array, columns=player_columns)
# player_df["batting_contribution"] = player_df.apply(lambda item: calculate_batting_contribution(item), axis=1)
# player_df["bowling_contribution"] = player_df.apply(lambda item: calculate_bowling_contribution(item), axis=1)
# player_df = player_df.loc[:, player_df.columns != "description"]
# player_df = player_df.loc[:, player_df.columns != "date"]
# player_df = player_df.loc[:, player_df.columns != "match_id"]
# player_df = player_df.loc[:, player_df.columns != "minutes_batted"]
# player_df = player_df.loc[:, player_df.columns != "score"]
# player_df = player_df.loc[:, player_df.columns != "rpo"]
# player_df = player_df.loc[:, player_df.columns != "target"]
# player_df = player_df.loc[:, player_df.columns != "extras"]
player_df.to_csv(output_file_encoded, index=False)
