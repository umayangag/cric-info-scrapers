from config.mysql import get_db_connection
import pandas as pd


def calculate_batting_consistency(db_connection):
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id, player_name FROM player")
    players_list = db_cursor.fetchall()

    for player in players_list:
        db_cursor.execute(
            f'SELECT description, runs, balls, minutes, fours, sixes, strike_rate FROM batting_data where player_id={player[0]}')
        player_data = db_cursor.fetchall()
        inning_count = len(player_data)
        if inning_count > 10:
            df = pd.DataFrame(player_data)
            not_out_count = len(df[df[0] == "not out"])
            sum_of_runs = df[1].sum()
            average_strike_rate = df[6].mean()
            average_score = sum_of_runs / (inning_count - not_out_count)

            centuries = len(df[df[1] >= 100])
            fifties = len(df[df[1] >= 50]) - centuries
            zeros = len(df[df[1] == 0])

            consistency = 0.4262 * average_score + 0.2566 * inning_count + 0.1510 * average_strike_rate + 0.0787 * centuries + 0.0556 * fifties - 0.0328 * zeros
            print(player[1], consistency)
            db_cursor.execute(f'UPDATE player SET batting_consistency = {consistency} WHERE player.id = {player[0]}')
        else:
            db_cursor.execute(f'UPDATE player SET batting_consistency = 0 WHERE player.id = {player[0]}')
    db_connection.commit()


db_connection = get_db_connection()
calculate_batting_consistency(db_connection)
