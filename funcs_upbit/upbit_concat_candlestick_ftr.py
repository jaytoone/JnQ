import pandas as pd
import os
import pickle
from datetime import datetime
from funcs.funcs_for_trade import consecutive_df, to_itvnum, itv_bn2ub, limit_by_itv

import pyupbit
import time

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 2500)


#           1 day = 86399000. (timestamp)       #
a_day = 3600 * 24 * 1000


def concat_candlestick(symbol, interval, days, limit=1500, end_date=None, show_process=False, timesleep=None):

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]

    startTime_ = datetime.timestamp(pd.to_datetime('{} 00:00:00'.format(end_date))) * 1000

    # if interval != '1m':
    #       trader 에서 자정 지나면 data 부족해지는 문제로 days >= 2 적용    #
    if interval == '1d':
        startTime_ -= a_day

    endTime = datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000

    if show_process:
        print(symbol)

    for day_cnt in range(days):

        if day_cnt != 0:
            startTime_ -= a_day
            endTime -= a_day

        # if show_process:
        #     print(datetime.fromtimestamp(startTime_ / 1000), end=' --> ')
        #     print(datetime.fromtimestamp(endTime / 1000))

        try:
            startTime = int(startTime_)
            endTime = int(endTime)

            df = pyupbit.get_ohlcv("{}".format(symbol),
                                   interval=itv_bn2ub(interval),
                                   count=limit,
                                   period=timesleep,
                                   to=datetime.fromtimestamp(endTime / 1000))

            if show_process:
                print(df.index[0], end=" --> ")
                print(df.index[-1])

            # print("endTime :", endTime)
            # print(df.head())
            # print(df.tail())
            # print(len(df))
            # print((df))
            # quit()

            assert len(df) != 0, "len(df) == 0"

            if day_cnt == 0:
                sum_df = df
            else:
                # print(df.head())
                # sum_df = pd.concat([sum_df, df])
                sum_df = df.append(sum_df)  # <-- -a_day 이기 때문에 sum_df 와 df 의 위치가 좌측과 같다.
                # print(sum_df)
                # quit()

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

    days = 300
    # days = 30

    end_date = '2021-07-01'
    end_date = '2021-11-25'
    # end_date = None

    # intervals = ['5m', '15m', '30m']
    intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '4h']
    # intervals = ['5m', '15m', '30m', '1h', '4h']

    concat_path = '../candlestick_concated/database_ub'

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]

    save_dir = os.path.join(concat_path, end_date)
    os.makedirs(save_dir, exist_ok=True)

    exist_files = os.listdir(save_dir)

    with open('../ticker_list/upbit_20211207.pkl', 'rb') as f:
        coin_list = pickle.load(f)

    coin_list.remove("KRW-ETH")
    print(coin_list)
    # quit()

    for coin in coin_list:

        for interval in intervals:

            # print(coin)

            #       check existing file     #
            #       Todo        #
            #        1. this phase require valid end_date       #

            save_name = '%s %s_%s.ftr' % (end_date, coin, interval)
            # print(exist_files)
            # print(save_name)
            # quit()

            if save_name in exist_files:
                print(save_name, 'exist !')
                continue

            limit = limit_by_itv(interval)

            try:
                concated_df, end_date = concat_candlestick(coin, interval, days,
                                                              end_date=end_date, limit=limit,
                                                              show_process=True, timesleep=0.05)
                # print("str(concated_df.index[-1]) :", str(concated_df.index[-1]))
                # quit()

            # try:

                verified_df = consecutive_df(concated_df, to_itvnum(interval))

                verified_df.reset_index().to_feather(os.path.join(save_dir, save_name), compression='lz4')

            except Exception as e:
                print('Error in save to_feather :', e)
                continue

            # print(concated_excel.tail())
            # print(len(concated_excel))
            # print(concated_excel.head())
            # quit()
            #
            # print('df[-1] timestamp :', datetime.timestamp(concated_excel.index[-1]))
            # print('current timestamp :', datetime.now().timestamp())
            # print(datetime.timestamp(concated_excel.index[-1]) < datetime.now().timestamp())
            # quit()

