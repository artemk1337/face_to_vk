import configparser
import os

import psycopg2

from settings import ROOT_PATH


DB_CONNECTOR_PATH = os.path.dirname(os.path.realpath(__file__))


""" <===== CONFIG =====> """


if os.path.exists(os.path.join(ROOT_PATH, 'database.conf')):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT_PATH, 'database.conf'))
else:
    raise ImportError("Can't import 'database.conf'")


""" <===== DB_CONNECTOR =====> """


db_cdf = cfg['database']

HOST = db_cdf['host']
PORT = db_cdf['port']
USERNAME = db_cdf['username']
PASSWORD = db_cdf['password']
DBNAME = db_cdf['dbname']

CONN = psycopg2.connect(
    host=HOST,
    port=PORT,
    user=USERNAME,
    password=PASSWORD,
    dbname=DBNAME
)


status_cfg = cfg['status']

STATUS_WAIT = status_cfg['wait']
STATUS_BUSY = status_cfg['busy']
STATUS_FINISH = status_cfg['finish']
STATUS_SET = (STATUS_WAIT, STATUS_BUSY, STATUS_FINISH)
