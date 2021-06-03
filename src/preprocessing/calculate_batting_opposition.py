from config.mysql import get_db_connection
import pandas as pd


def calculate_batting_opposition(db_connection):
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id, player_name FROM player")
    players_list = db_cursor.fetchall()
    db_cursor.execute("SELECT id, opposition_name FROM opposition")
    opposition_list = db_cursor.fetchall()

    for opposition in opposition_list:
        for player in players_list:
            db_cursor.execute(f'SELECT id FROM player_opposition_data WHERE player_id={player[0]} AND opposition_id={opposition[0]}')
            record_exist = len(db_cursor.fetchall())
            if record_exist == 0:
                db_cursor.execute(
                    f'INSERT INTO player_opposition_data SET player_id={player[0]}, opposition_id={opposition[0]}, batting_opposition=0, bowling_opposition=0')
    db_connection.commit()

    for opposition in opposition_list:
        for player in players_list:
            db_cursor.execute(
                f'SELECT description, runs, batting_data.balls, minutes, fours, sixes, strike_rate, '
                f'match_details.season_id FROM batting_data left join match_details '
                f'on batting_data.match_id=match_details.match_id where player_id = {player[0]} '
                f'and opposition_id = {opposition[0]};')
            player_data = db_cursor.fetchall()
            inning_count = len(player_data)
            if inning_count > 5:
                df = pd.DataFrame(player_data)
                not_out_count = len(df[df[0] == "not out"])
                sum_of_runs = df[1].sum()
                average_strike_rate = df[6].mean()
                if inning_count != not_out_count:
                    average_score = sum_of_runs / (inning_count - not_out_count)
                else:
                    average_score = sum_of_runs

                centuries = len(df[df[1] >= 100])
                fifties = len(df[df[1] >= 50]) - centuries
                zeros = len(df[df[1] == 0])

                form = 0.4262 * average_score + 0.2566 * inning_count + 0.1510 * average_strike_rate + 0.0787 * centuries + 0.0556 * fifties - 0.0328 * zeros
                print(player[1], form)
                db_cursor.execute(f'UPDATE player_opposition_data SET batting_opposition = {form} '
                                  f'WHERE player_id = {player[0]} AND opposition_id = {opposition[0]}')
    db_connection.commit()


db_connection = get_db_connection()
calculate_batting_opposition(db_connection)
