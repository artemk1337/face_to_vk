import configparser
import vk_api
import os

from settings import TMP_DIR


PLUGIN_PATH = os.path.dirname(os.path.realpath(__file__))


""" <===== CONFIG =====> """


if os.path.exists(os.path.join(PLUGIN_PATH, 'vk_parser.conf')):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(PLUGIN_PATH, 'vk_parser.conf'))
else:
    raise ImportError("Can't import 'vk_parser.conf'")


""" <===== VK_PARSER =====> """

# vk cookies
vk_cookies_filename = os.path.join(TMP_DIR, 'vk_config.json')

if 'vk_auth_account' in cfg:
    vk_auth_account_cfg = cfg['vk_auth_account']
    login = vk_auth_account_cfg['login']
    password = vk_auth_account_cfg['passowrd']

    VK_SESSION = vk_api.VkApi(login, password, config_filename=vk_cookies_filename)
    VK_SESSION.auth()
    VK_TOOLS = vk_api.VkTools(VK_SESSION)

elif 'vk_auth_bot' in cfg:
    vk_auth_bot_cfg = cfg['vk_auth_bot']
    app_id = int(vk_auth_bot_cfg['id'])
    secret_key = vk_auth_bot_cfg['secret_key']
    token = vk_auth_bot_cfg['service_key']

    VK_SESSION = vk_api.VkApi(app_id=app_id, client_secret=secret_key, token=token, config_filename=vk_cookies_filename)
    VK_TOOLS = vk_api.VkTools(VK_SESSION)

else:
    raise ImportError("Add 'auth_account' or 'auth_bot' in config")

# max buffer for iteration
ITER_MAX_BUFFER = cfg['vk_parser']['iter_max_objects']
# photo time
PHOTO_MAX_SHIFT_TIME = int(cfg['vk_parser']['photo_max_shift_time'])

# slow and fast parsers
PARSERS_ALL_DATA_SPEED = {
    'fast': VK_TOOLS.get_all,
    'slow': VK_TOOLS.get_all_slow,
}
PARSERS_ALL_DATA_ITER_SPEED = {
    'fast': VK_TOOLS.get_all_iter,
    'slow': VK_TOOLS.get_all_slow_iter,
}
speed_parser = cfg['vk_parser']['speed_parser']
if speed_parser not in ('slow', 'fast'):
    raise ImportError("Set vk speed_parser 'slow' or 'fast'")
PARSER_ALL = PARSERS_ALL_DATA_SPEED[speed_parser]
PARSER_ALL_ITER = PARSERS_ALL_DATA_ITER_SPEED[speed_parser]
