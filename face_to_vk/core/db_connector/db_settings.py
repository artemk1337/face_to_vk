import configparser
import psycopg2
import logging
import os

from settings import ROOT_PATH


DB_CONNECTOR_PATH = os.path.dirname(os.path.realpath(__file__))


""" <===== CONFIG =====> """


if os.path.exists(os.path.join(ROOT_PATH, 'face_to_vk.conf')):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT_PATH, 'face_to_vk.conf'))
else:
    raise ImportError("Can't import 'face_to_vk.conf'")


""" <===== DB_CONNECTOR =====> """


DB_LOGGER = logging.getLogger("db_connector")

db_cfg = dict(cfg)['database']
if not db_cfg:
    raise ImportError("Can't import database configs")

HOST = db_cfg['host']
PORT = db_cfg['port']
USERNAME = db_cfg['username']
PASSWORD = db_cfg['password']
DBNAME = db_cfg['dbname']

CONN = psycopg2.connect(
    host=HOST,
    port=PORT,
    user=USERNAME,
    password=PASSWORD,
    dbname=DBNAME
)

status_cfg = dict(cfg).get('db_status')
if not status_cfg:
    raise ImportError("Can't import database status configs")

STATUS_WAIT = status_cfg['wait']
STATUS_BUSY = status_cfg['busy']
STATUS_FINISH = status_cfg['finish']
STATUS_SET = (STATUS_WAIT, STATUS_BUSY, STATUS_FINISH)
