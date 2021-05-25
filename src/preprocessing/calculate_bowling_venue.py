from config.mysql import get_db_connection
import pandas as pd


def calculate_bowling_venue(db_connection):
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id, player_name FROM player")
    players_list = db_cursor.fetchall()
    db_cursor.execute("SELECT id, venue_name FROM venue")
    venue_list = db_cursor.fetchall()

    for venue in venue_list:
        for player in players_list:
            db_cursor.execute(
                f'SELECT bowling_data.balls, runs, bowling_data.wickets, econ, '
                f'match_details.venue_id FROM bowling_data left join match_details '
                f'on bowling_data.match_id=match_details.match_id where player_id = {player[0]} '
                f'and venue_id = {venue[0]};')
            player_data = db_cursor.fetchall()
            inning_count = len(player_data)
            if inning_count > 0:
                df = pd.DataFrame(player_data)
                total_overs = df[0].sum() / 6
                if df[2].sum() == 0:
                    strike_rate = df[0].sum()
                    average = df[1].sum()
                else:
                    strike_rate = df[0].sum() / df[2].sum()
                    average = df[1].sum() / df[2].sum()

                ff = len(df[df[2] >= 5])

                form = 0.3018 * total_overs + 0.2783 * inning_count + 0.1836 * strike_rate + 0.1391 * average + 0.0972 * ff

                print(player[1], venue)
                db_cursor.execute(f'UPDATE player_venue_data SET bowling_venue = {form} '
                                  f'WHERE player_id = {player[0]} AND venue_id = {venue[0]}')
    db_connection.commit()


db_connection = get_db_connection()
calculate_bowling_venue(db_connection)
