from easydict import EasyDict
import json

#       load config (refreshed by every trade)       #
dir_path = r"C:\Users\Lenovo\PycharmProjects\Project_System_Trading\JnQ\SE"
# with open(dir_path + '/config.json', 'r') as cfg:
with open(dir_path + '/config_v2.json', 'r') as cfg:
    config = EasyDict(json.load(cfg))

config.init_set.symbol_changed = False

with open(dir_path + '/config_v2_test.json', 'w') as cfg:
    json.dump(config, cfg, indent=2)