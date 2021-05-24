import mysql.connector

# mysql config
server_url = "localhost"
user = "root"
password = ""
auth_plugin = 'mysql_native_password'
database = "cricket_data"


def get_mysql_connection():
    return mysql.connector.connect(
        host=server_url,
        user=user,
        passwd=password,
        auth_plugin=auth_plugin,
    )


def get_db_connection():
    return mysql.connector.connect(
        host=server_url,
        user=user,
        passwd=password,
        auth_plugin=auth_plugin,
        database="cricket_data"
    )
