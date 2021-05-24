from config.mysql import get_db_connection
import pandas as pd


def calculate_batting_venue(db_connection):
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id, player_name FROM player")
    players_list = db_cursor.fetchall()
    db_cursor.execute("SELECT id, venue_name FROM venue")
    venue_list = db_cursor.fetchall()

    for venue in venue_list:
        for player in players_list:
            db_cursor.execute(f'INSERT INTO player_venue_data SET player_id={player[0]}, venue_id={venue[0]}')
    db_connection.commit()

    for venue in venue_list:
        for player in players_list:
            db_cursor.execute(
                f'SELECT description, runs, batting_data.balls, minutes, fours, sixes, strike_rate, '
                f'match_details.season_id FROM batting_data left join match_details '
                f'on batting_data.match_id=match_details.match_id where player_id = {player[0]} '
                f'and venue_id = {venue[0]};')
            player_data = db_cursor.fetchall()
            inning_count = len(player_data)
            if inning_count > 0:
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
                highest_score = df[1].max()

                form = 0.4262 * average_score + 0.2566 * inning_count + 0.1510 * average_strike_rate + 0.0787 * centuries + 0.0556 * fifties + 0.0328 * highest_score

                print(player[1], form)
                db_cursor.execute(f'UPDATE player_venue_data SET batting_venue = {form} '
                                  f'WHERE player_id = {player[0]} AND venue_id = {venue[0]}')
    db_connection.commit()


db_connection = get_db_connection()
calculate_batting_venue(db_connection)
