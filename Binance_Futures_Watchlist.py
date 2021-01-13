import pybithumb
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fake_useragent import UserAgent
import time
import os
import cv2
import Funcs_MACD_OSC
from finta import TA
from datetime import datetime
from scipy import stats

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
import numpy as np


future_coin = ['BTC', 'ETH', 'BCH', 'XRP', 'EOS', 'LTC', 'ETC',
               'LINK', 'XLM', 'ADA', 'XMR', 'SXP', 'KAVA', 'BAND',
               'DASH', 'ZEC', 'XTZ', 'BNB', 'ATOM', 'ONT', 'IOTA',
               'BAT', 'NEO', 'QTUM', 'WAVES', 'MKR', 'SNX', 'DOT',
               'THETA', 'ALGO', 'KNC', 'ZRX', 'COMP', 'OMG']

import pickle
with open('future_coin.p', 'wb') as f:
    pickle.dump(future_coin, f)

# with open('future_coin.p', 'rb') as f:
#     print(pickle.load(f))