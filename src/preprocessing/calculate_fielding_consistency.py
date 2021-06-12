from config.mysql import get_db_connection
import pandas as pd


def calculate_fielding_consistency(db_connection):
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id, player_name FROM player")
    players_list = db_cursor.fetchall()

    for player in players_list:
        db_cursor.execute(
            f'SELECT catches, run_outs, dropped_catches,missed_run_outs FROM fielding_data where player_id={player[0]}')
        player_data = db_cursor.fetchall()
        inning_count = len(player_data)
        if inning_count > 5:
            df = pd.DataFrame(player_data, columns=["catches", "run_outs", "dropped_catches", "missed_run_outs"])
            print(df)

            consistency = (df["catches"].sum() + df["run_outs"].sum()) / (
                    df["catches"].sum() + df["run_outs"].sum() + df["dropped_catches"].sum() +
                    df["missed_run_outs"].sum())
            print(player[1], consistency)
            db_cursor.execute(f'UPDATE player SET fielding_consistency = {consistency} WHERE player.id = {player[0]}')
        else:
            db_cursor.execute(f'UPDATE player SET fielding_consistency = 0 WHERE player.id = {player[0]}')
    db_connection.commit()


db_connection = get_db_connection()
calculate_fielding_consistency(db_connection)
