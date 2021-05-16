import csv
import mysql.connector
from datetime import datetime

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="root",
    auth_plugin='mysql_native_password',
    database="cricket_data"
)
db_cursor = db_connection.cursor()


def int_or_default(value):
    try:
        return int(value)
    except ValueError:
        return "NULL"


def float_or_default(value):
    try:
        return float(value)
    except ValueError:
        return "NULL"


def get_record_id(table, value):
    db_cursor.execute(f'SELECT id FROM {table} WHERE {table}_name LIKE "{value}"')
    records = db_cursor.fetchall()
    if db_cursor.rowcount == 0:
        db_cursor.execute(f'INSERT INTO {table} SET {table}_name="{value}"')
        db_connection.commit()
        db_cursor.execute(f'SELECT id FROM {table} WHERE {table}_name LIKE "{value}"')
        records = db_cursor.fetchall()
    return records[0][0]


def import_match_details():
    with open('data/match_details.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:
                opposition = row[8]
                oppositionId = get_record_id("opposition", opposition)

                venue = row[14]
                venueId = get_record_id("venue", venue)

                date_value = datetime.strptime(row[9], '%d-%b-%y')

                season = row[18]
                seasonId = get_record_id("season", season)

                toss = "lost"
                if row[17] == "TRUE":
                    toss = "won"

                match_number = row[19].replace("ODI No. ", "", 1)

                batting_session_time = int_or_default(row[12].replace(":", ".", 1).split(".", 1)[0])
                if batting_session_time == "NULL":
                    batting_session_label = ""
                elif batting_session_time >= 18:
                    batting_session_label = "night"
                else:
                    batting_session_label = "day"

                bowling_session_time = int_or_default(row[13].replace(":", ".", 1).split(".", 1)[0])
                if bowling_session_time == "NULL":
                    bowling_session_label = ""
                elif bowling_session_time >= 18:
                    bowling_session_label = "night"
                else:
                    bowling_session_label = "day"

                print(toss)

                db_cursor.execute(f'INSERT INTO match_details SET'
                                  f' score={int_or_default(row[0])},'
                                  f' wickets={int_or_default(row[1])},'
                                  f' overs={float_or_default(row[2])},'
                                  f' balls={int_or_default(row[3])},'
                                  f' rpo={float_or_default(row[4])},'
                                  f' target={int_or_default(row[5])},'
                                  f' inning="{"inning"+row[6]}",'
                                  f' result="{row[7]}",'
                                  f' opposition_id={oppositionId},'
                                  f' date="{date_value}",'
                                  f' match_id={int(row[10])},'
                                  f' batting_session="{batting_session_label}",'
                                  f' bowling_session="{bowling_session_label}",'
                                  f' venue_id="{venueId}",'
                                  f' extras={int_or_default(row[16])},'
                                  f' toss="{toss}",'
                                  f' season_id={seasonId}, '
                                  f' match_number={int(match_number)}'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')


def import_weather_data(filename, session):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:

                temp = row[5].replace("°c", "", 1)
                feels = row[6].replace("°c", "", 1)
                wind = row[7].split(" ", 1)[0]
                gust = row[8].split(" ", 1)[0]
                rain = row[9].split(" ", 1)[0]
                humidity = row[10].replace("%", "", 1)
                cloud = row[11].replace("%", "", 1)
                pressure = row[12].split(" ", 1)[0]
                viscosity = row[13].replace("°c", "", 1)

                print(line_count)
                db_cursor.execute(f'INSERT INTO weather_data SET'
                                  f' match_id={int(row[1])},'
                                  f' session="{session}",'
                                  f' temp={int_or_default(temp)},'
                                  f' feels={int_or_default(feels)},'
                                  f' wind={int_or_default(wind)},'
                                  f' gust={int_or_default(gust)},'
                                  f' rain={float_or_default(rain)},'
                                  f' humidity={int_or_default(humidity)},'
                                  f' cloud={int_or_default(cloud)},'
                                  f' pressure={int_or_default(pressure)},'
                                  f' viscosity="{viscosity}"'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')


def import_batting_data():
    with open('data/batting_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:
                player = row[0]
                playerId = get_record_id("player", player)

                print(line_count)
                db_cursor.execute(f'INSERT INTO batting_data SET'
                                  f' player_id={int(playerId)},'
                                  f' description="{row[1]}",'
                                  f' runs={int(row[2])},'
                                  f' balls={int(row[3])},'
                                  f' minutes={int_or_default(row[4])},'
                                  f' fours={int(row[5])},'
                                  f' sixes={int(row[6])},'
                                  f' strike_rate={float(row[7])},'
                                  f' batting_position={int(row[8])},'
                                  f' match_id={int(row[9])}'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')


def import_bowling_data():
    with open('data/bowling_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:
                player = row[0]
                playerId = get_record_id("player", player)

                print(line_count)
                db_cursor.execute(f'INSERT INTO bowling_data SET'
                                  f' player_id={int(playerId)},'
                                  f' overs={float(row[1])},'
                                  f' balls={int(row[2])},'
                                  f' maidens={int(row[3])},'
                                  f' runs={int(row[4])},'
                                  f' wickets={int(row[5])},'
                                  f' econ={float(row[6])},'
                                  f' dots={int(row[7])},'
                                  f' fours={int(row[8])},'
                                  f' sixes={int(row[9])},'
                                  f' wides={int(row[10])},'
                                  f' no_balls={int(row[11])},'
                                  f' match_id={int(row[12])}'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')


import_match_details()
import_weather_data('data/weather_data-batting.csv', "batting")
import_weather_data('data/weather_data-bowling.csv', "bowling")
import_batting_data()
import_bowling_data()
