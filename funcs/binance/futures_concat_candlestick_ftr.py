import pandas as pd
import os
from datetime import datetime
from binance_f import RequestClient
from binance_f.model import *
from binance_f.constant.test import *
from binance_f.base.printobject import *
import time

request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 2500)

#           1 day = 86399000. (timestamp)       #
a_day = 3600 * 24 * 1000
candle_limit = 1500


def concat_candlestick(symbol, interval, days, limit=candle_limit, end_date=None, show_process=False, timesleep=None):
    assert limit <= candle_limit, "assert limit < candle_limit"

    """ real_trading 에서 자정 지나면 data 부족해지는 문제로 assert, days >= 2 적용
    - imit under 1500 으로 cover 가능 (startTime = None 이면, 자정 이전 data load 함)
    - startTime 을 자정으로 설정해서 자정 이전의 data 를 가져오기 못한 것"""

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]
    startTime_ = datetime.timestamp(pd.to_datetime('{} 00:00:00'.format(end_date))) * 1000

    # if interval != '1m':
    if interval == '1d':
        startTime_ -= a_day

    endTime = datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000

    if show_process:
        print(symbol)

    if days > 1:  # 1일 이상의 data 가 필요한 경우 limit 없이 모두 가져옴
        limit = candle_limit

    # df_records = []
    for day_cnt in range(days):

        if day_cnt != 0:  # for loop 통해서 하루 이전 데이터를 지속적으로 가져옴
            startTime_ -= a_day
            endTime -= a_day

        try:
            # ------ limit < max_limit, startTime != None 일 경우, last_index 이상하게 (-> 어떻게..?) 나옴 ------ #
            if limit != candle_limit:
                startTime = None  # 변수명 get_candlestick_data 함수 param 이랑 일치시키기 위해 startTime.
            else:
                startTime = int(startTime_)
            df = request_client.get_candlestick_data(symbol=symbol,
                                                     interval=interval,
                                                     startTime=startTime, endTime=int(endTime), limit=limit)
            # ------ validation ------ #
            if show_process:
                print(df.index[0], end=" --> ")
                print(df.index[-1])
            assert len(df) != 0, "len(df) == 0"

            if day_cnt == 0:
                sum_df = df
            else:
                sum_df = pd.concat([df, sum_df])  # <-- -a_day 이기 때문에 sum_df 와 df 의 위치가 좌측과 같음.

            if timesleep is not None:
                time.sleep(timesleep)

        except Exception as e:
            print('error in get_candlestick_data :', e)

            if len(df) == 0:
                # quit()
                break

    # end_date = str(datetime.fromtimestamp(endTime / 1000)).split(' ')[0]
    # print(len(sum_df[~sum_df.index.duplicated(keep='first')]))

    # keep = 'last' 로 해야 중복기준 최신 df 를 넣는건데, 왜 first 로 해놓은거지
    return sum_df[~sum_df.index.duplicated(keep='last')], end_date


if __name__ == '__main__':

    days = 700  # 330 3
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
    symbol_list = [data.symbol for data in request_client.get_exchange_information().symbols if data.symbol not in ['BTCUSDT', 'ETHUSDT']]  # Binance all exchange symbol list
    # symbol_list = ['GALBUSD'] #, 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT', 'GMTUSDT', 'ADAUSDT']
    print(symbol_list)

    for symbol in symbol_list:
        for interval in intervals:
            #       check existing file     #
            #       Todo        #
            #        1. this phase require valid end_date       #
            save_name = '%s %s_%s.ftr' % (end_date, symbol, interval)
            # if save_name in exist_files:
            #     print(save_name, 'exist !')
            #     continue

            try:
                concated_df, end_date = concat_candlestick(symbol, interval, days, limit=1500,
                                                           end_date=end_date, show_process=True, timesleep=0.4)
                # print(concated_df.tail())
                # quit()
                concated_df.reset_index().to_feather(os.path.join(save_dir, save_name), compression='lz4')
            except Exception as e:
                print('Error in save to_excel :', e)
                continue
