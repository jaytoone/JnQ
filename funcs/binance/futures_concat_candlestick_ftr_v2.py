from binance.um_futures import UMFutures

import pandas as pd
import numpy as np
from datetime import datetime
import time
import pickle
import os


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 2500)

key_abspath = r"D:\Projects\System_Trading\JnQ\Bank\api_keys\binance_mademerich.pkl"
with open(key_abspath, 'rb') as f:
    api_key, secret_key = pickle.load(f)

um_futures_client = UMFutures(key=api_key, secret=secret_key)

# 1. a_day = 86399000. seconds (timestamp)
a_day = 3600 * 24 * 1000
candle_limit = 1500


def concat_candlestick(symbol, interval, days, limit, end_date=None, show_process=False, timesleep=None):

    """ 
    1. real_trade 에서 자정 지나면 data 부족해지는 문제로 assert, days >= 2 적용
        => imit under 1500 으로 cover 가능 (startTime = None 이면, 자정 이전 data load 함)
            = (startTime 을 자정으로 설정해서 자정 이전의 data 를 가져오기 못한 것)
    """

    assert limit <= candle_limit, "assert limit < candle_limit"

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]
        end_time = datetime.now().timestamp() * 1000
    # else:
    end_time = int(datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000)
    start_time_ = datetime.timestamp(pd.to_datetime('{} 00:00:00'.format(end_date))) * 1000

    # if interval != '1m':
    if interval == '1d':
        start_time_ -= a_day

    if show_process:
        print(symbol)

    if days > 1:  # 1일 이상의 data 가 필요한 경우 limit 없이 모두 가져옴
        limit = candle_limit

    df_list = []
    for day_cnt in range(days):

        if day_cnt != 0:  # for loop 통해서 하루 이전 데이터를 지속적으로 가져옴
            start_time_ -= a_day
            end_time -= a_day

        try:
            # a. limit < max_limit, startTime != None 일 경우, last_index 이상하게 (-> 어떻게..?) 나옴
            if limit != candle_limit:
                start_time = None  # 변수명 get_candlestick_data 함수 param 이랑 일치시키기 위해 start_time 사용.
            else:
                start_time = int(start_time_)

            response = um_futures_client.klines(symbol=symbol,
                                                interval=interval,
                                                startTime=start_time,
                                                endTime=int(end_time),
                                                limit=limit)

            df = pd.DataFrame(np.array(response)).set_index(0).iloc[:, :5]
            df.index = list(map(lambda x: datetime.fromtimestamp(int(x) / 1000), df.index))
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)

            # print(df.tail())

            # b. validation
            if len(df) == 0:
                # quit()
                print("len(df) == 0")
                break
            # assert len(df) != 0, "len(df) == 0"

            if show_process:
                print(df.index[0], end=" --> ")
                print(df.index[-1])

            df_list.append(df)
            # if day_cnt == 0:
            #     sum_df = df
            # else:
            #     sum_df = pd.concat([df, sum_df])  # <-- -a_day 이기 때문에 sum_df 와 df 의 위치가 좌측과 같음.

            if timesleep is not None:
                time.sleep(timesleep)

        except Exception as e:
            print("error in get_candlestick_data :", e)

    # keep = 'last' 로 해야 중복기준 최신 df 를 넣는건데, 왜 first 로 해놓은거지
    sum_df = pd.concat(df_list[::-1])

    return sum_df[~sum_df.index.duplicated(keep='last')], end_date


if __name__ == '__main__':

    days = 60  # 330
    end_date = None  # "2023-01-06" "2021-04-12" "2021-03-23"
    intervals = ['1m']  # ['1m', '3m', '5m', '15m', '30m', '1h', '4h'] - old

    concat_path = '../../database/binance/non_cum'

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]

    save_dir = os.path.join(concat_path, end_date)
    os.makedirs(save_dir, exist_ok=True)

    exist_files = os.listdir(save_dir)

    # with open('../symbol_list/binance_futures_20211207.pkl', 'rb') as f:
    #     symbol_list = pickle.load(f)
    # symbol_list = [data.symbol for data in request_client.get_exchange_information().symbols if data.symbol not in ['BTCUSDT', 'ETHUSDT']]  # Binance all exchange symbol list
    symbol_list = ['ETHUSDT']  #, 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT', 'GMTUSDT', 'ADAUSDT']
    print(symbol_list)

    for symbol in symbol_list:
        for interval in intervals:

            # 1. check existing file
            #       a. this phase require valid end_date       #
            save_name = '%s %s_%s.ftr' % (end_date, symbol, interval)
            # if save_name in exist_files:
            #     print(save_name, 'exist !')
            #     continue

            try:
                concated_df, end_date = concat_candlestick(symbol, interval, days,
                                                           limit=1500,
                                                           end_date=None,
                                                           show_process=True,
                                                           timesleep=0.2)
                # print(concated_df.iloc[[0, -1]].dtypes)
                # print(concated_df.open.iloc[-1])
                # print(concated_df.tail())
                # print(type(concated_df.open.iloc[-1]))
                # quit()
                concated_df.reset_index().to_feather(os.path.join(save_dir, save_name), compression='lz4')
            except Exception as e:
                print("error in save to_excel : {}".format(e))
                continue
