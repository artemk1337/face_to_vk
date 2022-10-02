import vk_api
import configparser
import logging
import os


# set root dir
HOMEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(HOMEDIR)


LOG_FILENAME = os.path.join(HOMEDIR, 'logs/face_to_vk.txt')
os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILENAME
)
logging.getLogger("face_to_vk")


""" <===== CONFIG =====> """


if os.path.exists('face_to_vk.conf'):
    cfg = configparser.ConfigParser()
    cfg.read('face_to_vk.conf')
else:
    raise ImportError("Can't import 'face_to_vk.conf'")


# dirs
TMP_DIR = cfg['dirs']['tmp_dir']
DATA_DIR = cfg['dirs']['data_dir']
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


""" <===== VK PLUGIN =====> """


# vk parser cookies
vk_cookies_filename = os.path.join(TMP_DIR, 'vk_config.json')

if 'vk_auth_account' in cfg:
    vk_auth_account_cfg = cfg['vk_auth_account']
    login = vk_auth_account_cfg['login']
    password = vk_auth_account_cfg['passowrd']

    VK_SESSION = vk_api.VkApi(login, password, config_filename=vk_cookies_filename)
    VK_SESSION.auth()
    TOOLS = vk_api.VkTools(VK_SESSION)

elif 'vk_auth_bot' in cfg:
    vk_auth_bot_cfg = cfg['vk_auth_bot']
    app_id = int(vk_auth_bot_cfg['id'])
    secret_key = vk_auth_bot_cfg['secret_key']
    token = vk_auth_bot_cfg['service_key']

    VK_SESSION = vk_api.VkApi(app_id=app_id, client_secret=secret_key, token=token, config_filename=vk_cookies_filename)
    TOOLS = vk_api.VkTools(VK_SESSION)

else:
    raise ImportError("Add 'auth_account' or 'auth_bot' in config")

# max buffer for iteration
ITER_MAX_BUFFER = cfg['vk_parser']['iter_max_objects']
# photo time
PHOTO_MAX_SHIFT_TIME = int(cfg['vk_parser']['photo_max_shift_time'])

# slow and fast parsers
PARSERS_ALL_DATA_BY_SPEED = {
    'fast': TOOLS.get_all,
    'slow': TOOLS.get_all_slow,
}
speed_parser = cfg['vk_parser']['speed_parser']
if speed_parser not in ('slow', 'fast'):
    raise ImportError("Set vk speed_parser 'slow' or 'fast'")
PARSER_ALL = PARSERS_ALL_DATA_BY_SPEED[speed_parser]
