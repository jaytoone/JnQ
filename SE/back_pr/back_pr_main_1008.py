import os
from SE.back_pr.back_pr_strat.v4_1 import back_pr_check
import pandas as pd
import pickle
from SE.utils.v4_1 import *
from datetime import datetime
from easydict import EasyDict
import json

#       본, logic 은 partial_tp 를 지원하는 것으로 알고있음       #

#       Todo        #
#        1. change upper dir
#        2. back_pr_check
#         a. remove key variable
#         b. add dir_path as an instance
#        3. custom modification (ex. sync_check)

dir_path = os.path.abspath('./../')
print("dir_path :", dir_path)
# quit()
os.chdir("./../..")
print("os.getcwd() :", os.getcwd())

# quit()


if __name__ == "__main__":

    # ----- below phase considered import location ----- #
    from binance_futures_concat_candlestick import concat_candlestick

    date = "2021-06-30"

    #       load config (refreshed by every trade)       #
    with open(dir_path + '/config_v2.json', 'r') as cfg:
        config = EasyDict(json.load(cfg))

    init_set = config.init_set
    ep_set = config.ep_set
    out_set = config.out_set
    tp_set = config.tp_set
    lvrg_set = config.lvrg_set

    show_log = False
    history = False
    # print("os.getcwd() :", os.getcwd())

    # ----------- history ver. ----------- #
    # res_df_name = "candlestick_concated/res_df/%s %s_trix_backi2.xlsx" % (date, init_set.symbol)
    # res_df_name = "candlestick_concated/res_df/2021-07-01 ETHUSDT_backi2.xlsx"
    # res_df_name = "candlestick_concated/res_df/2021-07-01 ADAUSDT_majorst_backi2.xlsx"

    dict_name = "2021-07-01 ETHUSDT_cbline_backi2_res_dfs.pkl"

    #     load with pickle    #
    with open("candlestick_concated/res_df/" + dict_name, 'rb') as f:
        res_df_dict = pickle.load(f)

    print(dict_name, "loaded !")
    history = True

    # ----------- trade log ver. ----------- #
    log_name = "1633611106.pkl"

    #        1. trade_log 의 start ~ end index 만큼 일단 res_df 잘라야함     #
    #        2. log 의 open & close data 를 사용해, back_pr 수익과 비교해야함        #
    #        2-1. trade_log 만큼 잘라진 res_df 에, open & close data 를 삽입     #
    #       Todo        #
    #        3. plot_check 도 고려     #
    if not history:

        # with open("basic_v1/trade_log/" + log_name, "rb") as dict_f:
        with open(dir_path + "/trade_log/" + log_name, "rb") as dict_f:
            trade_log = pickle.load(dict_f)
            # print(trade_log)
            # print(list(trade_log.items())[1:])
            # quit()

        start_timestamp = int(log_name.split(".")[0])
        start_datetime = datetime.fromtimestamp(start_timestamp)
        # quit()

        start_datetime = pd.to_datetime(str(start_datetime)[:-2] + "59.999000")
        end_datetime = pd.to_datetime(trade_log['last_trading_time'][:17] + "59.999000")

        print("start_datetime :", start_datetime)
        print("end_datetime :", end_datetime)

        end_date = str(end_datetime).split(" ")[0]
        # print("end_date :", end_date)
        # quit()

        #        days 계산 해야함        #
        end_timestamp = datetime.timestamp(end_datetime)
        days = int((end_timestamp - start_timestamp) / (3600 * 24)) + 2
        days_2 = days + 2
        print("days :", days)
        # quit()

        #       1. concat_candle ver.      #
        new_df, _ = concat_candlestick(init_set.symbol, init_set.interval, days=days, end_date=end_date, timesleep=0.2, show_process=True)

        # print(type(new_df.index[0]))

        new_df2, _ = concat_candlestick(init_set.symbol, init_set.interval2, days=days_2, end_date=end_date, timesleep=0.2, show_process=True)
        new_df3, _ = concat_candlestick(init_set.symbol, init_set.interval3, days=days_2, end_date=end_date, timesleep=0.2, show_process=True)

        #       1-1. load saved_df ver. => for pr logic confirmation         #
        # new_df = pd.read_excel("candlestick_concated/%s/%s %s.xlsx" % (init_set.interval, date, init_set.symbol), index_col=0)
        # new_df2 = pd.read_excel("candlestick_concated/%s/%s %s.xlsx" % (init_set.interval2, date, init_set.symbol), index_col=0)

        res_df_ = sync_check(new_df, new_df2, new_df3)

        #        back_pr index range 를 선택       #
        # new_df_real = new_df
        # new_df_real = new_df.loc[start_datetime:]
        res_df = res_df_.loc[start_datetime:end_datetime].copy()
        print("sync_check phase done !")

        # print(new_df_real.tail())
        # quit()

    #       3. directly load res_df     #
    if history:
        for key, res_df_ in res_df_dict.items():
            if init_set.symbol in key:
                res_df = res_df_
                break

        res_df = st_level(res_df, '5m', 0.5)
        print("res_df from dict phase done !")

    #       Todo        #
    #        adjust proper strategy     #
    #        check pr        #
    ep_tp_list, trade_list = back_pr_check(res_df, dir_path, lvrg_set.leverage)

    if not history:

        #       insert open & close data to res_df --> plot_check 를 고려하면, res_df 에 삽입하는게 좋을것         "
        res_df['back_ep'] = np.nan
        res_df['back_tp'] = np.nan
        for ep_tp, t_ij in zip(ep_tp_list, trade_list):

            for ep_i, ep_idx in enumerate(t_ij[0]):
                if show_log:
                    print("ep_idx :", ep_idx)
                    print("ep_tp[0][ep_i] :", ep_tp[0][ep_i])
                res_df['back_ep'].iloc[ep_idx] = ep_tp[0][ep_i]

            for tp_i, tp_idx in enumerate(t_ij[1]):
                if show_log:
                    print("tp_idx :", tp_idx)
                    print("ep_tp[1][tp_i] :", ep_tp[1][tp_i])
                res_df['back_tp'].iloc[tp_idx] = ep_tp[1][tp_i]

        # print()
        # quit()

        #       insert real-trade data      #
        res_df['real_ep'] = np.nan
        res_df['real_tp'] = np.nan
        for t_k, t_v in list(trade_log.items())[1:]:
            if len(t_v) == 3:
                res_df['real_ep'].loc[pd.to_datetime(t_k)] = t_v[0]
            else:
                res_df['real_tp'].loc[pd.to_datetime(t_k)] = t_v[0]

        #       span 이 길면 xlsx 저장이 어려울 수 있음 (오래걸림)      #
        # res_df.to_excel("basic_v1/back_pr/back_pr.xlsx", index=True)
        res_df.to_excel(dir_path + "/back_pr/back_pr.xlsx", index=True)

        print("back_pr.xlsx saved !")



