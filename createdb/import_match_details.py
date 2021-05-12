import csv
import mysql.connector

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
        return 0


def float_or_default(value):
    try:
        return float(value)
    except ValueError:
        return 0


def get_record_id(table, value):
    db_cursor.execute(f'SELECT id FROM {table} WHERE {table}_name LIKE "{value}"')
    records = db_cursor.fetchall()
    if db_cursor.rowcount == 0:
        db_cursor.execute(f'INSERT INTO {table} SET {table}_name="{value}"')
        db_connection.commit()
        db_cursor.execute(f'SELECT id FROM {table} WHERE {table}_name LIKE "{value}"')
        records = db_cursor.fetchall()
    return records[0][0]


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

            season = row[18]
            seasonId = get_record_id("season", season)

            toss = 0
            if row[17] == "TRUE":
                toss = 1

            match_number = row[19].replace("ODI No. ", "", 1)

            print(line_count)
            db_cursor.execute(f'INSERT INTO match_details SET'
                              f' score={int_or_default(row[0])},'
                              f' wickets={int_or_default(row[1])},'
                              f' overs={float_or_default(row[2])},'
                              f' balls={int_or_default(row[3])},'
                              f' rpo={float_or_default(row[4])},'
                              f' target={int_or_default(row[5])},'
                              f' inning={int_or_default(row[6])},'
                              f' result="{row[7]}",'
                              f' opposition_id={oppositionId},'
                              f' date="{row[9]}",'
                              f' match_id={int(row[10])},'
                              f' batting_session="{row[12]}",'
                              f' bowling_session="{row[13]}",'
                              f' venue_id="{venueId}",'
                              f' extras={int_or_default(row[16])},'
                              f' toss={toss},'
                              f' season_id={seasonId}, '
                              f' match_number={int(match_number)}'
                              f'')
        line_count += 1
    db_connection.commit()
    print(f'Processed {line_count} lines.')
