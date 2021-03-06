from config.mysql import get_db_connection
import pandas as pd


def calculate_bowling_form(db_connection):
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id, player_name FROM player")
    players_list = db_cursor.fetchall()
    db_cursor.execute("SELECT id, season_name FROM season")
    season_list = db_cursor.fetchall()

    for season in season_list:
        for player in players_list:
            db_cursor.execute(f'SELECT id FROM player_form_data WHERE player_id={player[0]} AND season_id={season[0]}')
            record_exist = len(db_cursor.fetchall())
            if record_exist == 0:
                db_cursor.execute(
                    f'INSERT INTO player_form_data SET player_id={player[0]}, season_id={season[0]}, batting_form=0, bowling_form=0')
    db_connection.commit()

    for season in season_list:
        for player in players_list:
            season_id = season[0]
            if season_id > 1:
                season_id = season_id - 1
            db_cursor.execute(
                f'SELECT bowling_data.balls, runs, bowling_data.wickets, econ, '
                f'match_details.season_id FROM bowling_data left join match_details '
                f'on bowling_data.match_id=match_details.match_id where player_id = {player[0]} '
                f'and season_id = {season_id};')
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

                form = 0.3269 * total_overs + 0.2846 * inning_count + 0.1877 * strike_rate + 0.1210 * average + 0.0798 * ff
                print(player[1], form)
                db_cursor.execute(f'UPDATE player_form_data SET bowling_form = {form} '
                                  f'WHERE player_id = {player[0]} AND season_id = {season[0]}')
    db_connection.commit()


db_connection = get_db_connection()
calculate_bowling_form(db_connection)
