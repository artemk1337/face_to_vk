import configparser
import os


DB_CONNECTOR_PATH = os.path.dirname(os.path.realpath(__file__))


""" <===== CONFIG =====> """


if os.path.exists(os.path.join(DB_CONNECTOR_PATH, 'database.conf')):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(DB_CONNECTOR_PATH, 'database.conf'))
else:
    raise ImportError("Can't import 'database.conf'")


""" <===== DB_CONNECTOR =====> """


db_cdf = cfg['database']

HOST = db_cdf['host']
PORT = db_cdf['port']
USERNAME = db_cdf['username']
PASSWORD = db_cdf['password']
DBNAME = db_cdf['dbname']


status_cfg = cfg['status']

STATUS_WAIT = status_cfg['wait']
STATUS_BUSY = status_cfg['busy']
STATUS_FINISH = status_cfg['finish']
STATUS_SET = (STATUS_WAIT, STATUS_BUSY, STATUS_FINISH)
