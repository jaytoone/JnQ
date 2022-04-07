import os
# import pandas as pd
import pickle
import importlib
from datetime import datetime
from easydict import EasyDict
from funcs.olds.funcs_indicator_candlescore import *
import json


if __name__ == "__main__":

    cur_path = os.getcwd()
    dir_path = os.path.dirname(cur_path)
    print("dir_path :", dir_path)

    #        Todo          #
    #         input params ------- #
    strat_pkg = 'AT'
    frame_ver = "0114_htf"
    ID_list = ['v3', 'v5_2', 'v7_3']

    show_log = 0
    save_ftr = 0  # default 1, save sync_checked res_df when history=0
    history = 1  # if you don't want to use trade_log
    save_xlsx = 0  # save back ep & tp, saved_ftr should be exist

    #       save xlsx var.      #
    new_cols = ['open', 'high', 'low', 'close']

    #       history variables       #
    # date = "2021-11-17"
    date = "2022-01-10"
    ticker = "ETHUSDT"
    ftr_path = "../candlestick_concated/res_df/concat/cum/{}/{} {}.ftr".format(date, date, ticker)
    # else:

    # ----------- trade log ver. ----------- #
    log_name = "ETHUSDT_1642038440.pkl"

    # ------------- link module ------------- #
    bot_name = "{}.bots.{}_bot_{}_{}_{}".format(strat_pkg, frame_ver, *ID_list)
    bot_lib = importlib.import_module(bot_name)

    strat_name = "{}.back_pr.back_id.{}_back_id_{}_{}_{}".format(strat_pkg, frame_ver, *ID_list)
    strat_lib = importlib.import_module(strat_name)

    # ----- below phase considered import location ----- # -> 위에서 cwd 가 변경되는 이유로 바로 import 가능해짐
    from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick

    #   history=1, 에서도 save_xlsx 을 위해 public 으로 선언
    trade_log_path = os.path.join(dir_path, "trade_log", bot_lib.trader_name.split(".")[-1])

    cfg_path_list = [os.path.join(dir_path, "config", name_) for name_ in bot_lib.config_list]
    cfg_file_list = [open(cfg_path) for cfg_path in cfg_path_list]
    cfg_list = [EasyDict(json.load(cfg_)) for cfg_ in cfg_file_list]
    [config1, config2, config3] = cfg_list
    #       opened files should be closed --> 닫지 않으면 reopen 시 error occurs         #
    _ = [cfg_.close() for cfg_ in cfg_file_list]

    config = cfg_list[0]

    save_path = os.path.join(dir_path, "back_pr", "back_res", bot_lib.trader_name.split(".")[-1])
    try:
        os.makedirs(save_path, exist_ok=True)
    except Exception as e:
        print("error in makedirs for logger :", e)

    #        1. trade_log 의 start ~ end index 만큼 일단 res_df 잘라야함     #
    #        2. log 의 open & close data 를 사용해, back_pr 수익과 비교해야함        #
    #        2-1. trade_log 만큼 잘라진 res_df 에, open & close data 를 삽입     #
    #       Todo        #
    #        3_2. plot_check 도 고려     #
    if not history:

        # with open(trade_log_path + "/%s" % log_name, "rb") as dict_f:
        with open(os.path.join(trade_log_path, log_name, "rb")) as dict_f:
            trade_log = pickle.load(dict_f)

        start_timestamp = int(log_name.split("_")[-1].split(".")[0])
        start_datetime = datetime.fromtimestamp(start_timestamp)
        # quit()

        start_datetime = pd.to_datetime(str(start_datetime)[:-2] + "59.999000")
        end_datetime = pd.to_datetime(trade_log['last_trading_time'][:17] + "59.999000")

        #       Todo        #
        #        4. temporary datetime     #
        # start_datetime = pd.to_datetime(str("2022-01-10 00:49:59.999000"))
        # end_datetime = pd.to_datetime(str("2022-01-10 14:30:59.999000"))
        # start_timestamp = datetime.timestamp(start_datetime)

        print("start_datetime :", start_datetime)
        print("end_datetime :", end_datetime)

        end_date = str(end_datetime).split(" ")[0]
        save_ftr_dir = "../candlestick_concated/res_df/concat/cum/{}".format(end_date)
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

        res_df_, _ = concat_candlestick(config.trader_set.symbol, '1m',
                                        days=days_2,
                                        end_date=end_date,
                                        limit=1500,
                                        timesleep=0.2,
                                        show_process=1)

        res_df_ = bot_lib.utils_public_lib.sync_check(res_df_, config, skip_row=True)
        # print(res_df_.index[0])

        #        back_pr index range 를 선택       #
        res_df = res_df_.loc[start_datetime:end_datetime].copy()
        print("sync_check phase done !")

        if save_ftr:
            try:
                os.makedirs(save_ftr_dir, exist_ok=True)
            except Exception as e:
                print("error in makedirs for logger :", e)

            res_df.reset_index().to_feather(os.path.join(save_ftr_dir, save_ftr_name), compression='lz4')
            print("{} saved !".format(save_ftr_name))

        # print(res_df.index[0])
        # quit()

    #       3. directly load res_df     #
    else:  # just for back_strat's validity

        res_df = pd.read_feather(ftr_path, columns=None, use_threads=True).set_index("index")

        print(ftr_path, "loaded !")

    if history:
        png_name = "{} {} back_pr.png".format(date, ticker)
    else:
        png_name = "{} {} back_pr.png".format(end_date, config.trader_set.symbol)

    png_save_path = os.path.join(save_path, png_name)
    open_list, ep_tp_list, trade_list, side_list, strat_ver_list = \
        strat_lib.back_pr_check(res_df, png_save_path, bot_lib.utils_public_lib, bot_lib.utils_list, cfg_list, config.trader_set.symbol)

    # if not history and save_xlsx:
    if save_xlsx:

        #       save with new cols      #
        res_df_save = res_df[new_cols].copy()

        #       insert open & close data to res_df --> plot_check 를 고려하면, res_df 에 삽입하는게 좋을것         "
        res_df_save['back_ep_init'] = np.nan
        res_df_save['back_ep'] = np.nan
        res_df_save['back_tp'] = np.nan
        for ep_init, ep_tp, t_ij, side, strat_ver in zip(open_list, ep_tp_list, trade_list, side_list, strat_ver_list):

            res_df_save['back_ep_init'].iloc[ep_init] = "ep_init_{}_{}".format(side, strat_ver)
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
        #       log_name
        res_df_save.to_excel(save_path + "/{} {} back_pr.xlsx".format(date, ticker), index=True)

        print("back_pr.xlsx saved !")

    print("system proc. done")
    quit()



