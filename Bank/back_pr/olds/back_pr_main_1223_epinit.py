import os
# import pandas as pd
import pickle
import importlib
from datetime import datetime
from easydict import EasyDict
from funcs.olds.funcs_indicator_candlescore import *
import json


if __name__ == "__main__":

    #       Todo       #
    #        1. match bot name      #
    from Banking.bots.bot_v3n5_2_1222_emgncy import trader_name, utils_list, config_list

    dir_path = os.getcwd()
    # print("os.getcwd() :", os.getcwd())

    strat_name = "AT.back_pr.back_id.back_v3n5_2.py".replace(".py", "")
    strat_lib = importlib.import_module(strat_name)

    # ----- below phase considered import location ----- # -> 위에서 cwd 가 변경되는 이유로 바로 import 가능해짐
    from funcs.binance.futures_concat_candlestick_ftr import concat_candlestick

    #        Todo       #
    #         1. set input params
    show_log = 0
    save_ftr = 1    # save sync_checked res_df when history=0
    save_xlsx = 1   # save back ep & tp
    history = 1     # if you don't want to use trade_log
    if history:
        # date = "2022-11-17"
        date = "2022-01-01"
        ftr_path = "../database/res_df/concat/cum/{}/{} ETHUSDT.ftr".format(date, date)
    else:
        log_name = "ETHUSDT_1640263052.pkl"

    config1_name, config2_name = config_list
    utils_public_lib, utils1_lib, utils2_lib = utils_list

    save_path = os.path.join(dir_path, "back_pr", "back_res", trader_name.split(".")[-1])
    trade_log_path = os.path.join(dir_path, "trade_log", trader_name.split(".")[-1])
    cfg_full_path1 = os.path.join(dir_path, "config", config1_name)
    cfg_full_path2 = os.path.join(dir_path, "config", config2_name)

    try:
        os.makedirs(save_path, exist_ok=True)
    except Exception as e:
        print("error in makedirs for logger :", e)

    #       load config (refreshed by every trade)       #
    with open(cfg_full_path1) as cfg:
        config1 = EasyDict(json.load(cfg))
    with open(cfg_full_path2) as cfg:
        config2 = EasyDict(json.load(cfg))

    config_list = [config1, config2]
    #       Todo        #
    #        3. single back_pr
    # config_list = [config2, config1]
    # utils_list = utils_public_lib, utils2_lib, utils1_lib

    config = config_list[0]

    #        1. trade_log 의 start ~ end index 만큼 일단 res_df 잘라야함     #
    #        2. log 의 open & close data 를 사용해, back_pr 수익과 비교해야함        #
    #        2-1. trade_log 만큼 잘라진 res_df 에, open & close data 를 삽입     #
    #       Todo        #
    #        3. plot_check 도 고려     #
    if not history:

        # ----------- trade log ver. ----------- #
        with open(trade_log_path + "/%s" % log_name, "rb") as dict_f:
            trade_log = pickle.load(dict_f)
            # print(trade_log)
            # print(list(trade_log.items())[1:])
            # quit()

        start_timestamp = int(log_name.split("_")[-1].split(".")[0])
        start_datetime = datetime.fromtimestamp(start_timestamp)
        # quit()

        start_datetime = pd.to_datetime(str(start_datetime)[:-2] + "59.999000")
        end_datetime = pd.to_datetime(trade_log['last_trading_time'][:17] + "59.999000")

        #       Todo        #
        #        1. temporary datetime     #
        # start_datetime = pd.to_datetime(str("2021-12-01 15:49:59.999000"))
        # end_datetime = pd.to_datetime(str("2021-12-31 17:49:59.999000"))
        # start_timestamp = datetime.timestamp(start_datetime)

        print("start_datetime :", start_datetime)
        print("end_datetime :", end_datetime)

        end_date = str(end_datetime).split(" ")[0]
        save_ftr_dir = "../database/res_df/concat/cum/{}".format(end_date)
        save_ftr_name = "{} {}.ftr".format(end_date, config.trader_set.symbol)

        # print("end_date :", end_date)
        # quit()

        #        days 계산 해야함        #
        end_timestamp = datetime.timestamp(end_datetime)
        days = int((end_timestamp - start_timestamp) / (3600 * 24)) + 2
        days_2 = days + 2
        print("days :", days)
        print()
        # quit()

        #       1. concat_candle ver.      #
        res_df_list = []
        day_list = [days, days_2, days_2]
        for itv_i, interval_ in enumerate(config.trader_set.interval_list):

            if interval_ != "None":

                new_df_, _ = concat_candlestick(config.trader_set.symbol, interval_,
                                                days=day_list[itv_i],
                                                end_date=end_date,
                                                limit=1500,
                                                timesleep=0.2,
                                                show_process=True)
                # if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                #     old_df = 1
                #     break
                # else:
                new_df = new_df_
            else:
                new_df = None

            res_df_list.append(new_df)

        res_df_ = utils_public_lib.sync_check(res_df_list)
        if save_ftr:
            try:
                os.makedirs(save_ftr_dir, exist_ok=True)
            except Exception as e:
                print("error in makedirs for logger :", e)
            res_df_.reset_index().to_feather(os.path.join(save_ftr_dir, save_ftr_name), compression='lz4')
            print("{} saved !".format(save_ftr_name))

        # print(res_df_.index[0])

        #        back_pr index range 를 선택       #
        # new_df_real = new_df
        # new_df_real = new_df.loc[start_datetime:]
        res_df = res_df_.loc[start_datetime:end_datetime].copy()
        print("sync_check phase done !")

        # print(res_df.index[0])
        # quit()

    #       3. directly load res_df     #
    else:  # just for back_strat's validity

        res_df = pd.read_feather(ftr_path, columns=None, use_threads=True).set_index("index")

        print(ftr_path, "loaded !")

    open_list, ep_tp_list, trade_list = strat_lib.back_pr_check(res_df, save_path, config_list, utils_list, config.trader_set.symbol)

    # if not history and save_xlsx:
    if save_xlsx:

        #       save with new cols      #
        new_cols = ['open', 'high', 'low', 'close']
        res_df_save = res_df[new_cols].copy()

        #       insert open & close data to res_df --> plot_check 를 고려하면, res_df 에 삽입하는게 좋을것         "
        res_df_save['back_ep_init'] = np.nan
        res_df_save['back_ep'] = np.nan
        res_df_save['back_tp'] = np.nan
        for ep_init, ep_tp, t_ij in zip(open_list, ep_tp_list, trade_list):

            res_df_save['back_ep_init'].iloc[ep_init] = "ep_init"
            ep_idx_list, out_idx_list, tp_idx_list = t_ij   # out_idx_list 추가됨 (e_j 를 보관할 list 필요함)

            for ep_i, ep_idx in enumerate(ep_idx_list):
                if show_log:
                    print("ep_idx :", ep_idx)
                    print("ep_tp[0][ep_i] :", ep_tp[0][ep_i])
                res_df_save['back_ep'].iloc[ep_idx] = ep_tp[0][ep_i]

            for tp_i, tp_idx in enumerate(tp_idx_list):
                if show_log:
                    print("tp_idx :", tp_idx)
                    print("ep_tp[1][tp_i] :", ep_tp[1][tp_i])
                res_df_save['back_tp'].iloc[tp_idx] = ep_tp[1][tp_i]

        # print()
        # quit()

        #       insert real-trade data      #
        res_df_save['real_ep_init'] = np.nan
        res_df_save['real_ep'] = np.nan
        res_df_save['real_tp'] = np.nan
        for t_k, t_v in list(trade_log.items())[1:]:

            if t_v[-1] == "init_open":
                res_df_save['real_ep_init'].loc[pd.to_datetime(t_k)] = t_v[0]
            elif t_v[-1] == "open":
                res_df_save['real_ep'].loc[pd.to_datetime(t_k)] = t_v[0]
            else:
                res_df_save['real_tp'].loc[pd.to_datetime(t_k)] = t_v[0]

        #       span 이 길면 xlsx 저장이 어려울 수 있음 (오래걸림)      #
        res_df_save.to_excel(save_path + "/back_pr.xlsx", index=True)

        print("back_pr.xlsx saved !")

    print("system proc. done")
    quit()



