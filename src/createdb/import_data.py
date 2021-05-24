from config.mysql import get_db_connection
from shared.utils import *
from importers.import_match_details import import_match_details
from importers.import_weather_data import import_weather_data
from importers.import_batting_data import import_batting_data
from importers.import_bowling_data import import_bowling_data
import os

dirname = os.path.dirname(__file__)
match_details_filepath = os.path.join(dirname, 'data\\match_details.csv')
weather_data_batting_filepath = os.path.join(dirname, 'data\\weather_data-batting.csv')
weather_data_bowling_filepath = os.path.join(dirname, 'data\\weather_data-bowling.csv')
batting_data_filepath = os.path.join(dirname, 'data\\batting_data.csv')
bowling_data_filepath = os.path.join(dirname, 'data\\bowling_data.csv')

db_connection = get_db_connection()

import_match_details(db_connection, match_details_filepath)
import_weather_data(db_connection, weather_data_batting_filepath, "batting")
import_weather_data(db_connection, weather_data_bowling_filepath, "bowling")
import_batting_data(db_connection, batting_data_filepath)
import_bowling_data(db_connection, bowling_data_filepath)
