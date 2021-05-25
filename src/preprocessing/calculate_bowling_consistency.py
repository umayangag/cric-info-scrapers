from config.mysql import get_db_connection
import pandas as pd


def calculate_bowling_consistency(db_connection):
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id, player_name FROM player")
    players_list = db_cursor.fetchall()

    for player in players_list:
        db_cursor.execute(
            f'SELECT balls, runs, wickets, econ FROM bowling_data where player_id={player[0]}')
        player_data = db_cursor.fetchall()
        inning_count = len(player_data)
        if inning_count > 5:
            df = pd.DataFrame(player_data)
            total_overs = df[0].sum() / 6
            strike_rate = df[0].sum() / df[2].sum()
            average = df[1].sum() / df[2].sum()
            ff = len(df[df[2] >= 5])

            consistency = 0.4174 * total_overs + 0.2634 * inning_count + 0.1602 * strike_rate + 0.0975 * average + 0.0615 * ff
            print(player[1], consistency)
            db_cursor.execute(f'UPDATE player SET bowling_consistency = {consistency} WHERE player.id = {player[0]}')
        else:
            db_cursor.execute(f'UPDATE player SET bowling_consistency = {0} WHERE player.id = {player[0]}')
    db_connection.commit()


db_connection = get_db_connection()
calculate_bowling_consistency(db_connection)
