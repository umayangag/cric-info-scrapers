import csv

from mysql.connector import utils
from .config import  get_db_connection
from datetime import datetime
from .shared.utils import *
from .importers.import_match_details import  import_match_details
from .importers.import_weather_data import import_weather_data
from .importers.import_batting_data import import_batting_data
from .importers.import_bowling_data import import_bowling_data
import os

dirname = os.path.dirname(__file__)
match_details_filepath = os.path.join(dirname, 'data\\match_details.csv')
weather_data_batting_filepath = os.path.join(dirname, 'data\\weather_data-batting.csv')
weather_data_bowling_filepath = os.path.join(dirname, 'data\\weather_data-bowling.csv')
batting_data_filepath = os.path.join(dirname, 'data\\batting_data.csv')
bowling_data_filepath = os.path.join(dirname, 'data\\bowling_data.csv')

db_connection = get_db_connection()
db_cursor = db_connection.cursor()

import_match_details(match_details_filepath)
import_weather_data(weather_data_batting_filepath, "batting")
import_weather_data(weather_data_bowling_filepath, "bowling")
import_batting_data(batting_data_filepath)
import_bowling_data(bowling_data_filepath)
