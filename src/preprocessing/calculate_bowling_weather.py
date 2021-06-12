from config.mysql import get_db_connection
import pandas as pd


def calculate_bowling_weather(db_connection):
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id, player_name FROM player")
    players_list = db_cursor.fetchall()

    for weather_class in range(1, 11):
        for player in players_list:
            db_cursor.execute(f'SELECT id FROM player_weather_data WHERE player_id={player[0]} AND weather_category={weather_class}')
            record_exist = len(db_cursor.fetchall())
            if record_exist == 0:
                db_cursor.execute(
                    f'INSERT INTO player_weather_data SET player_id={player[0]}, weather_category={weather_class}, bowling_weather=0, bowling_weather=0')
    db_connection.commit()

    for weather_class in range(1,11):
        for player in players_list:
            db_cursor.execute(
                f'SELECT bowling_data.balls, runs, bowling_data.wickets, econ, '
                f'weather_category FROM bowling_data left join weather_data '
                f'on bowling_data.match_id=weather_data.match_id where player_id = {player[0]} '
                f'and weather_category = {weather_class};')
            player_data = db_cursor.fetchall()
            inning_count = len(player_data)
            if inning_count > 5:
                df = pd.DataFrame(player_data)
                total_overs = df[0].sum() / 6
                if df[2].sum() == 0:
                    strike_rate = df[0].sum()
                    average = df[1].sum()
                else:
                    strike_rate = df[0].sum() / df[2].sum()
                    average = df[1].sum() / df[2].sum()

                ff = len(df[df[2] >= 5])

                weather = 0.3269 * total_overs + 0.2846 * inning_count + 0.1877 * strike_rate + 0.1210 * average + 0.0798 * ff
                print(player[1], weather)
                db_cursor.execute(f'UPDATE player_weather_data SET bowling_weather = {weather} '
                                  f'WHERE player_id = {player[0]} AND weather_category={weather_class}')
    db_connection.commit()


db_connection = get_db_connection()
calculate_bowling_weather(db_connection)
