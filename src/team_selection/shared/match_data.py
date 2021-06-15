from config.mysql import get_db_connection

db_connection = get_db_connection()
db_cursor = db_connection.cursor()


def get_match_ids():
    db_cursor.execute(
        f'SELECT match_id FROM match_details')
    results = db_cursor.fetchall()
    return results


def get_fielding_data(match_id, player_id):
    db_cursor.execute(
        f'SELECT catches, run_outs, dropped_catches, missed_run_outs '
        f'FROM fielding_data WHERE match_id={match_id} and player_id={player_id}')
    results = db_cursor.fetchall()
    if len(results) == 1:
        return results[0]
    else:
        return 0, 0, 0, 0


def get_batting_data(match_id, player_id):
    db_cursor.execute(
        f'SELECT description, runs, balls, fours, sixes, strike_rate, batting_position '
        f'FROM batting_data WHERE match_id={match_id} and player_id={player_id}')
    results = db_cursor.fetchall()
    if len(results) == 1:
        return results[0]
    else:
        return 0, 0, 0, 0, 0, 0, 0


def get_bowling_data(match_id, player_id):
    db_cursor.execute(
        f'SELECT runs, balls, wickets, econ '
        f'FROM bowling_data WHERE match_id={match_id} and player_id={player_id}')
    results = db_cursor.fetchall()
    if len(results) == 1:
        return results[0]
    else:
        return 0, 0, 0


def get_match_data(match_id, label):
    db_cursor.execute(
        f'SELECT inning, {label}_session, toss,venue_id, opposition_id, season_id,score, wickets, balls, target, extras, match_id, result '
        f'FROM match_details WHERE match_id={match_id} ')
    return db_cursor.fetchall()[0]


def get_weather_data(match_id, label):
    db_cursor.execute(
        f'SELECT temp,wind, rain, humidity, cloud, pressure, viscosity '
        f'FROM weather_data WHERE match_id={match_id} and session LIKE "{label}"')

    return db_cursor.fetchall()[0]


def get_player_metric(match_id, label1, player_obj, label2, label3, id):
    search_id = id
    if id == 0:
        search_id = 1
    db_cursor.execute(
        f'SELECT {label1}_{label2} FROM player_{label2}_data WHERE player_id={player_obj[0]} and {label3}_id={search_id}')
    data_list = db_cursor.fetchall()
    if len(data_list) > 0:
        value = data_list[0][0]
        if value > 0:
            return value
    return player_obj[f'{label1}_consistency']


def get_player_list(match_id, label):
    db_cursor.execute(
        f'SELECT player.id, player.player_name, player.is_wicket_keeper, player.is_retired, player.batting_consistency, '
        f'player.bowling_consistency, player.fielding_consistency FROM {label}_data '
        f'left join player on player.id={label}_data.player_id '
        f'WHERE match_id = "{match_id}"')
    return db_cursor.fetchall()
