from binance_f.model import *
import json
from easydict import EasyDict
from datetime import datetime
import pandas as pd


if __name__ == '__main__':

    patch_start_time = '2021-03-12 14:58:00'
    patch_end_time = '2021-03-12 16:58:00'
    timestamp_PST = datetime.timestamp(pd.to_datetime('{}'.format(patch_start_time)))
    timestamp_PET = datetime.timestamp(pd.to_datetime('{}'.format(patch_end_time)))

    while 1:

        current_time = datetime.now()
        # print(current_time.timestamp())

        # print(timestamp_PST)
        # print(current_time.timestamp() > timestamp_PST)
        # quit()

        if current_time.timestamp() > timestamp_PST:
            with open('futures_bot_config.json', 'r') as cfg:
                config = EasyDict(json.load(cfg))

            # json.dump(config, cfg, indent=1)
            # print(json.dumps(config, indent=1))

            #       shut down arima_bot     #
            config['FUNDMTL']['run'] = False
            with open('futures_bot_config.json', 'w') as cfg:
                # json.load()
                json.dump(config, cfg, indent=1)

            while 1:

                current_time = datetime.now()

                if current_time.timestamp() > timestamp_PET:
                    with open('futures_bot_config.json', 'r') as cfg:
                        config = EasyDict(json.load(cfg))

                    # json.dump(config, cfg, indent=1)
                    # print(json.dumps(config, indent=1))

                    #       turn on arima_bot     #
                    config['FUNDMTL']['run'] = True
                    with open('futures_bot_config.json', 'w') as cfg:
                        # json.load()
                        json.dump(config, cfg, indent=1)
                    quit()
