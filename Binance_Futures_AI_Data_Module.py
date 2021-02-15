# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# import scipy.misc
# import itertools
import random
from binance_futures_modules import *
# from Funcs_For_Trade import *
from Funcs_Indicator import *
from Funcs_for_TP_Ratio_TVT_Bias_Binarize_modified import profitage
from binance_futures_concat_candlestick import concat_candlestick
from Make_X_TP_Ratio_Realtime_for_Bot import made_x
from easydict import EasyDict
import time


seed = 1218
np.random.seed(seed)
random.seed(seed)

# model_range = range(1180, 1181)

model_init = True
model_renew = False
while 1:

    time.sleep(.5)

    #            Configuration             #
    with open('binance_futures_bot_config.json', 'r') as cfg:
        config = EasyDict(json.load(cfg))

    fundamental = config.FUNDMTL
    ai = config.AI

    if ai.long_thr_precision is None and ai.short_thr_precision is None:
        model_life = 3 * 60
    else:
        model_life = ai.model_life

    if model_init:
        model_init = False
        model_renew_time = time.time()
        model_renew = True
    else:
        if time.time() - model_renew_time > model_life:
            model_renew = True
            model_renew_time = time.time()
        else:
            time.sleep(model_life - (time.time() - model_renew_time))
            continue

    model_range = [ai.model_num]

    if model_renew:
        model_renew = False

        for model_num in model_range:

            # try:

            first_df, _ = concat_candlestick(fundamental.symbol, '1m', ai.train_days)
            second_df, _ = concat_candlestick(fundamental.symbol, '3m', ai.train_days)
            third_df, _ = concat_candlestick(fundamental.symbol, '30m', ai.train_days)

            df = profitage(first_df, second_df, third_df, label_type=LabelType.TRAIN, show_time=False)
            print('train df.index[-1] :', df.index[-1])

            Made_X, Made_Y, _ = made_x(df, ai.input_data_length, ai.scale_window_size, ai.train_data_amount, label_type=LabelType.TRAIN)

            np.save('Made_X/Made_X %s.npy' % model_num, Made_X)
            np.save('Made_X/Made_Y %s.npy' % model_num, Made_Y)





