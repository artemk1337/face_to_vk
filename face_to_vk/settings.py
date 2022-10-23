import configparser
import logging
import os


# set root dir
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# os.chdir(HOMEDIR)


""" <===== COMMON CONFIG =====> """


if os.path.exists(os.path.join(ROOT_PATH, 'face_to_vk.conf')):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT_PATH, 'face_to_vk.conf'))
else:
    raise ImportError("Can't import 'face_to_vk.conf'")


# dirs
dirs_cfg = dict(cfg).get('dirs')
if not dirs_cfg:
    raise ImportError("Can't import dirs configs")

TMP_DIR = os.path.join(ROOT_PATH, dirs_cfg['tmp_dir'])
DATA_DIR = os.path.join(ROOT_PATH, dirs_cfg['data_dir'])
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


""" <===== LOGGER CONFIG =====> """


logger_cfg = dict(cfg).get('logger')
if not logger_cfg:
    raise ImportError("Can't import logger configs")

level = logger_cfg['level']
filename = os.path.join(ROOT_PATH, logger_cfg['filename'])

os.makedirs(os.path.dirname(filename), exist_ok=True)
logging.basicConfig(
    filename=filename,
    level=level
)

MAIN_LOGGER = logging.getLogger("face_to_vk")
