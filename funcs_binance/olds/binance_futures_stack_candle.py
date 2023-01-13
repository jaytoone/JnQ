import pandas as pd
from binance_f.constant.test import *
from easydict import EasyDict
from binance_futures_concat_candlestick import concat_candlestick
import json


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 2500)


if __name__ == '__main__':

    interval = '30m'
    concat_type = 'new'
    # symbol = 'DOTUSDT'

    # with open('future_coin.p', 'rb') as f:
    #     coin_list = pickle.load(f)
    coin_list = ['BTC']

    for coin in coin_list:

        symbol = coin + 'USDT'

        #       concat date exist       #
        try:
            df1 = pd.read_excel('database/%s/2021-05-07 %s.xlsx' % (interval, symbol), index_col=0)
            df2 = pd.read_excel('database/%s/2021-05-17 %s.xlsx' % (interval, symbol), index_col=0)

            df3 = df1.append(df2)
            df3 = df3[~df3.index.duplicated(keep='last')]
            df3.to_excel('database/%s/2021-05-17 %s.xlsx' % (interval, symbol))

            print(symbol, 'concatenated !')

        except Exception as e:
            print('Error in read_excel :', e)