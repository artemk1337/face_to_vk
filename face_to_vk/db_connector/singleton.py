import psycopg2

from db_connector.settings import HOST, PORT, USERNAME, PASSWORD, DBNAME


CONN = psycopg2.connect(
    host=HOST,
    port=PORT,
    user=USERNAME,
    password=PASSWORD,
    dbname=DBNAME
)


