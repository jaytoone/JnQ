import os
from back_pr_funcs import cloud_lb_pr
import pandas as pd
import pickle
from basic_v1.utils_v2 import *
from datetime import datetime

os.chdir("./../..")

from binance_futures_concat_candlestick import concat_candlestick


if __name__ == "__main__":

    symbol = "ETHUSDT"
    interval = '1m'
    interval2 = '3m'

    date = "2021-06-30"
    res_df_name = "candlestick_concated/res_df/%s %s_trix_backi2.xlsx" % (date, symbol)

    log_name = "1627023773.pkl"

    #        1. trade_log 의 start ~ end index 만큼 일단 res_df 잘라야함     #
    #        2. log 의 open & close data 를 사용해, back_pr 수익과 비교해야함        #
    #        2-1. trade_log 만큼 잘라진 res_df 에, open & close data 를 삽입     #
    #       Todo        #
    #        3. plot_check 도 고려     #

    with open("basic_v1/trade_log/" + log_name, "rb") as dict_f:
        trade_log = pickle.load(dict_f)
        # print(trade_log)
        # print(list(trade_log.items())[1:])
        # quit()

    start_timestamp = int(log_name.split(".")[0])
    start_datetime = datetime.fromtimestamp(start_timestamp)
    # quit()
    start_datetime = pd.to_datetime(str(start_datetime).replace(str(start_datetime)[-2:], "59.999000"))
    end_datetime = pd.to_datetime(trade_log['last_trading_time'].replace(trade_log['last_trading_time'][17:], "59.999000"))

    print("start_datetime :", start_datetime)
    # print("end_datetime :", end_datetime)
    # quit()

    #       1. concat_candle ver.      #
    new_df, _ = concat_candlestick(symbol, interval, days=1, timesleep=0.2)

    # print(type(new_df.index[0]))

    #       new_df 와 new_df2 의 sync 를 맞추기 위해서 end_datetime 은 생략        #
    #       Todo        #
    #        back_pr index range 를 선택       #
    new_df_real = new_df
    # new_df_real = new_df.loc[start_datetime:]

    # print(new_df_real.tail())
    # quit()

    new_df2, _ = concat_candlestick(symbol, interval2, days=1, timesleep=0.2)

    #       1-1. load saved_df ver. => for pr logic confirmation         #
    # new_df = pd.read_excel("candlestick_concated/%s/%s %s.xlsx" % (interval, date, symbol), index_col=0)
    # new_df2 = pd.read_excel("candlestick_concated/%s/%s %s.xlsx" % (interval2, date, symbol), index_col=0)

    res_df = sync_check(new_df_real, new_df2, cloud_on=True)

    #       3. directly load res_df     #
    # res_df = pd.read_excel(res_df_name, index_col=0)
    # print("sync_check phase done !")

    #       check pr        #
    ep_tp_list, trade_list = cloud_lb_pr(res_df)

    #       insert open & close data to res_df --> plot_check 를 고려하면, res_df 에 삽입하는게 좋을것         "
    res_df['back_ep'] = np.nan
    res_df['back_tp'] = np.nan
    for ep_tp, t_ij in zip(ep_tp_list, trade_list):
        res_df['back_ep'].iloc[t_ij[0]] = ep_tp[0]
        res_df['back_tp'].iloc[t_ij[1]] = ep_tp[1]

    #       insert real-trade data      #
    res_df['real_ep'] = np.nan
    res_df['real_tp'] = np.nan
    for t_k, t_v in list(trade_log.items())[1:]:
        if len(t_v) == 3:
            res_df['real_ep'].loc[pd.to_datetime(t_k)] = t_v[0]
        else:
            res_df['real_tp'].loc[pd.to_datetime(t_k)] = t_v[0]

    res_df.to_excel("basic_v1/back_pr/back_pr.xlsx", index=True)

    print("back_pr.xlsx saved !")



