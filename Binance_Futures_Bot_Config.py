from binance_f import RequestClient
from binance_f.constant.test import *
from binance_f.base.printobject import *
from binance_f.model.constant import *
import json

request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)


class LabelType:
    OPEN = 'Open'
    CLOSE = 'Close'
    PROFIT = 'Profit'
    TRAIN = 'Train'
    TEST = 'Test'


config_json = dict()

config_json["FUNDMTL"] = {
    'run': True,
    'symbol': ["ALGOUSDT"],
    'bar_close_second': 59,  # <-- in 1m bar
    'accumulated_income': 0.0,
    'accumulated_profit': 1.0,
}

config_json["ORDER"] = {
    'entry_type': OrderType.LIMIT,
    'entry_execution_wait': 2 * 60,   # if you use limit entry
    'exit_execution_wait': 1 * 60,
    'breakout_qty_ratio': 2 / 3,  # if you use limit entry
    'tp_type': OrderType.LIMIT,
    'sl_type': OrderType.LIMIT,
}

config_json["AI"] = {
    'model_life': 1200 * 60,
    'train_module_file': 'Binance_Futures_AI_Train_Module.py',
    'data_module_file': 'Binance_Futures_AI_Data_Module.py',
    'train_days': 7,
    'test_days': 4,
    'a_day': 3600 * 24 * 1000,
    'scale_window_size': 3000,
    'input_data_length': 30,
    'model_num': 1190,
    'sl_least_gap_ratio': 0.1,
    'target_precision': 0.9,
    'long_thr_precision': None,
    'short_thr_precision': None,
    'train_data_amount': 6000,
}

config_json["AI"]['test_data_amount'] = config_json["AI"]['scale_window_size']


if __name__=='__main__':

    print(config_json)

    with open('Binance_Futures_Bot_Config.json', 'w') as cfg:
        json.dump(config_json, cfg)
