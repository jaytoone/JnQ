import time

import pandas as pd
import os
import pickle
from datetime import datetime
from funcs.public.broker import itv_binance_to_upbit, limit_by_itv

import pyupbit
# import time

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 2500)


#           1 day = 86399000. (timestamp)       #
a_day = 3600 * 24 * 1000
candle_limit = 1500


def concat_candlestick(symbol, interval, days, limit=candle_limit, end_date=None, timesleep=None, show_process=False):
    assert limit <= candle_limit, "assert limit < candle_limit"

    """ 
    1. real_trade 에서 자정 지나면 data 부족해지는 문제로 assert, days >= 2 적용
        => imit under 1500 으로 cover 가능 (startTime = None 이면, 자정 이전 data load 함)
            = (startTime 을 자정으로 설정해서 자정 이전의 data 를 가져오기 못한 것)
    """

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]
    startTime_ = datetime.timestamp(pd.to_datetime('{} 00:00:00'.format(end_date))) * 1000

    # if interval != '1m':
    if interval == '1d':
        startTime_ -= a_day

    endTime = datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000

    if show_process:
        print(symbol)

    # if days > 1:  # 1일 이상의 data 가 필요한 경우 limit 없이 모두 가져옴
    #     limit = candle_limit

    # df_records = []
    # for day_cnt in range(days):
    day_cnt = 0
    while 1:

        # 1. days = None --> 모든 data 추출.
        if days is not None:
            if day_cnt >= days:
                break

        # a. loop 통해서 하루 이전 데이터를 지속적으로 가져옴
        if day_cnt != 0:
            startTime_ -= a_day
            endTime -= a_day

        try:
            # b.  limit < max_limit, startTime != None 일 경우, last_index 이상하게 (-> 어떻게..?) 나옴 ------ #
            # if limit != candle_limit:
            #     startTime = None  # 변수명 get_candlestick_data 함수 param 이랑 일치시키기 위해 startTime.
            # else:
            #     startTime = int(startTime_)

            df = pyupbit.get_ohlcv("{}".format(symbol),
                                   interval=itv_binance_to_upbit(interval),
                                   count=limit,
                                   period=timesleep,
                                   to=datetime.fromtimestamp(endTime / 1000))

            #     i. if count <= 200, timesleep not adj to get_ohlcv()
            if limit <= 200:
                # print("limit :", limit)
                time.sleep(timesleep)

            # c. validation
            #       i. 더이상 추출할 데이터가 없는 경우 break.
            if df is None:
                print("no more data.")
                break

            if show_process:
                print(df.index[0], end=" ~ ")
                print(df.index[-1])
            assert len(df) != 0, "len(df) == 0"

            if day_cnt == 0:
                sum_df = df
            else:
                sum_df = pd.concat([df, sum_df])  # <-- -a_day 이기 때문에 sum_df 와 df 의 위치가 좌측과 같음.

        except Exception as e:
            print('error in get_candlestick_data :', e)
            break
        else:
            day_cnt += 1

    # keep = 'last' 로 해야 중복기준 최신 df 를 넣는건데, 왜 first 로 해놓은건지.
    return sum_df[~sum_df.index.duplicated(keep='last')], end_date


if __name__ == '__main__':

    days = 10  # 330 3

    """
    upbit 에서 당일 기준 이전 데이터를 조회할지도 None 은 성립하지 않는다.
    """

    end_date = None  # None 2023-07-04 # "2023-01-06" "2021-04-12" "2021-03-23"
    intervals = ['D']  # ['1m', '3m', '5m', '15m', '30m', '1h', '4h'] - old

    concat_path = r'D:\Projects\System_Trading\JnQ\database\upbit'

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]

    save_dir = os.path.join(concat_path, end_date)
    os.makedirs(save_dir, exist_ok=True)

    exist_files = os.listdir(save_dir)

    from pyupbit import get_tickers
    # with open(r'D:\Projects\System_Trading\JnQ\database\ticker_list\upbit_20230718.pkl', 'wb') as f:
    #     pickle.dump(get_tickers(), f)
    with open(r'D:\Projects\System_Trading\JnQ\database\ticker_list\upbit_20230718.pkl', 'rb') as f:
        symbol_list = pickle.load(f)

    # exist_files = os.listdir(os.path.join(concat_path, "2023-03-23"))
    # exist_symbols = list(map(lambda x: x.replace("2023-03-23 ", "").replace("_1m.ftr", ""), exist_files))
    # get_tickers
    # print(len([symbol_ for symbol_ in get_tickers() if 'KRW' in symbol_]))
    # quit()
    # symbol_list = ['KRW-CBK', 'KRW-CHZ', 'KRW-FCT2', 'KRW-IOST', 'KRW-MBL', 'KRW-SHIB', 'KRW-STPT']
    symbol_list = ['KRW-MBL']

    for symbol in symbol_list:
        for interval in intervals:
            #       check existing file     #
            #       Todo        #
            #        1. this phase require valid end_date       #
            save_name = "{} {}_{}.ftr".format(end_date, symbol, interval)

            if save_name in exist_files:
                print(save_name, 'exist !')
                continue

            # if symbol in exist_symbols or 'KRW' not in symbol_list:
            #     print("skip {}".format(symbol))
            #     continue

            limit = limit_by_itv(interval)
            try:
                concated_df, end_date = concat_candlestick(symbol, interval, days, limit=limit,
                                                           end_date=end_date, show_process=True, timesleep=0.11)
                quit()
                save_path = os.path.join(save_dir, save_name)
                concated_df.reset_index().to_feather(save_path, compression='lz4')
                print(save_path, "saved.\n")
            except Exception as e:
                print("error in save to_feather :", e)
                continue

