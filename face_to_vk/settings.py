import configparser
import logging
import os


# set root dir
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# os.chdir(HOMEDIR)


""" <===== LOGGING =====> """


LOG_FILENAME = os.path.join(ROOT_PATH, 'logs/face_to_vk.txt')
os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILENAME,
    level='INFO'
)
LOGGER = logging.getLogger("face_to_vk")


""" <===== CONFIG =====> """


if os.path.exists(os.path.join(ROOT_PATH, 'face_to_vk.conf')):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT_PATH, 'face_to_vk.conf'))
else:
    raise ImportError("Can't import 'face_to_vk.conf'")


# dirs
TMP_DIR = os.path.join(ROOT_PATH, cfg['dirs']['tmp_dir'])
DATA_DIR = os.path.join(ROOT_PATH, cfg['dirs']['data_dir'])
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
