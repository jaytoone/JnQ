import os

pkg_path = os.path.abspath('./../')
os.chdir(pkg_path)

from funcs_binance.binance_futures_modules import *
from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick
from funcs_binance.funcs_order_logger import limit_order, partial_limit_v2, market_close_order
from funcs.funcs_trader import intmin
import numpy as np  # for np.nan
import time
import pickle
import logging.config
from pathlib import Path
from easydict import EasyDict
from datetime import datetime
import shutil


class Trader:

    def __init__(self, utils_list, config_list):

        #           static platform variables        #
        self.utils_public, self.utils1, self.utils2 = utils_list
        self.config1_name, self.config2_name = config_list
        self.utils = None

        self.over_balance = None
        self.min_balance = 5.0  # USDT

        self.accumulated_income = 0.0
        self.calc_accumulated_profit = 1.0
        self.accumulated_profit = 1.0
        self.accumulated_back_profit = 1.0

        self.sub_client = None

    def run(self):

        exec_file_name = Path(__file__).stem

        sys_log_path = os.path.join(pkg_path, "sys_log", exec_file_name)
        trade_log_path = os.path.join(pkg_path, "trade_log", exec_file_name)
        df_log_path = os.path.join(pkg_path, "df_log", exec_file_name)

        os.makedirs(sys_log_path, exist_ok=True)
        os.makedirs(trade_log_path, exist_ok=True)
        os.makedirs(df_log_path, exist_ok=True)

        cfg_full_path1 = os.path.join(pkg_path, "config", self.config1_name)
        cfg_full_path2 = os.path.join(pkg_path, "config", self.config2_name)
        initial_run = 1
        while 1:

            #       load config (refreshed by every trade)       #
            try:
                with open(cfg_full_path1, 'r') as cfg:
                    config1 = EasyDict(json.load(cfg))
                with open(cfg_full_path2, 'r') as cfg:
                    config2 = EasyDict(json.load(cfg))

                #       open 전 base_config 는 config1 로 설정       #
                config2.trader_set = config1.trader_set
                config = config1

            except Exception as e:
                print("error in load config :", e)
                time.sleep(1)
                continue

            if config.trader_set.symbol_changed:
                initial_run = 1

                #       1. rewrite symbol_changed to false
                config.trader_set.symbol_changed = 0

                with open(cfg_full_path1, 'w') as cfg:
                    json.dump(config, cfg, indent=2)

            if initial_run:

                trade_log = {}
                trade_log_name = "{}_{}.pkl".format(config.trader_set.symbol,
                                                    str(datetime.now().timestamp()).split(".")[0])

                trade_log_fullpath = os.path.join(trade_log_path, trade_log_name)
                src_cfg_path = sys_log_path.replace(exec_file_name, "base_cfg.json")
                # dst_cfg_path = os.path.join(sys_log_path, trade_log_name.replace(".pkl", ".json"))
                dst_cfg_path = os.path.join(sys_log_path, "{}.json".format(config.trader_set.symbol))
                # print("dst_cfg_path :", dst_cfg_path)
                # quit()

                try:
                    shutil.copy(src_cfg_path, dst_cfg_path)

                except Exception as e:
                    print("error in shutil.copy() :", e)
                    continue

            try:
                with open(dst_cfg_path, 'r') as sys_cfg:
                    sys_log_cfg = EasyDict(json.load(sys_cfg))

                    if initial_run:
                        # sys_log_cfg.handlers.file_rot.filename = dst_cfg_path.replace(".json", ".log")
                        sys_log_cfg.handlers.file_rot.filename = \
                            os.path.join(sys_log_path, trade_log_name.replace(".pkl", ".log"))
                        logging.getLogger("apscheduler.executors.default").propagate = False

                        with open(dst_cfg_path, 'w') as edited_cfg:
                            json.dump(sys_log_cfg, edited_cfg, indent=3)

                    logging.config.dictConfig(sys_log_cfg)
                    sys_log = logging.getLogger()

            except Exception as e:
                print("error in load sys_log_cfg :", e)
                time.sleep(config.trader_set.api_retry_term)
                continue

            if initial_run:

                sys_log.info('# ----------- {} ----------- #'.format(exec_file_name))
                sys_log.info("pkg_path : {}".format(pkg_path))

                while 1:  # for completion
                    # --- 1. leverage type => "isolated" --- #
                    try:
                        request_client.change_margin_type(symbol=config.trader_set.symbol,
                                                          marginType=FuturesMarginType.ISOLATED)
                    except Exception as e:
                        sys_log.error('error in change_margin_type : {}'.format(e))
                    else:
                        sys_log.info('leverage type --> isolated')

                    # --- 2. confirm limit leverage --- #
                    try:
                        limit_leverage = get_limit_leverage(symbol_=config.trader_set.symbol)
                    except Exception as e:
                        sys_log.error('error in get limit_leverage : {}'.format(e))
                        continue
                    else:
                        sys_log.info('limit_leverage : {}'.format(limit_leverage))

                    # --- 3. sub_client --- #
                    try:
                        sub_client.subscribe_aggregate_trade_event(config.trader_set.symbol.lower(), callback, error)
                        self.sub_client = sub_client
                    except Exception as e:
                        sys_log.error('error in get sub_client : {}'.format(e))
                        continue

                    break

                initial_run = 0
                sys_log.info("initial_set done\n")

            #       check run        #
            if not config.trader_set.run:
                time.sleep(config.trader_set.realtime_term)  # give enough time to read config
                continue

            #       get startTime     #
            start_timestamp = int(time.time() * 1000)

            #       initial param.      #
            self.utils = None
            open_side = None
            ep_loc_check = 1
            ep_loc_pointcnt = 0
            ep_out = 0
            load_new_df = 1

            while 1:

                if ep_loc_check:

                    #       log last trading time     #
                    #       미체결을 고려해, ep_loc_check 마다 log 수행       #
                    trade_log["last_trading_time"] = str(datetime.now())
                    with open(trade_log_fullpath, "wb") as dict_f:
                        pickle.dump(trade_log, dict_f)

                    temp_time = time.time()

                    if load_new_df:

                        try:

                            res_df_list = []
                            old_df = 0
                            for itv_i, interval_ in enumerate(config.trader_set.interval_list):

                                if interval_ != "None":

                                    new_df_, _ = concat_candlestick(config.trader_set.symbol, interval_,
                                                                    days=1,
                                                                    limit=config.trader_set.row_list[itv_i],
                                                                    timesleep=0.2,
                                                                    show_process=0)
                                    if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                        old_df = 1
                                        break
                                    else:
                                        new_df = new_df_.iloc[-config.trader_set.row_list[itv_i]:].copy()
                                else:
                                    new_df = None

                                res_df_list.append(new_df)

                            if old_df:
                                continue

                            sys_log.info("complete load_df execution time : {}".format(datetime.now()))

                        except Exception as e:
                            sys_log.error("error in load new_dfs : {}".format(e))
                            time.sleep(config.trader_set.api_retry_term)
                            continue

                        else:
                            load_new_df = 0

                    sys_log.info("~ load_new_df time : %.2f" % (time.time() - temp_time))
                    # temp_time = time.time()

                    # ------------ get indi.  ------------ #
                    try:
                        res_df = self.utils_public.sync_check(res_df_list)  # function usage format maintenance
                        sys_log.info('~ sync_check time : %.5f' % (time.time() - temp_time))

                        res_df = self.utils_public.public_indi(res_df)
                        sys_log.info('~ public_indi time : %.5f' % (time.time() - temp_time))

                    except Exception as e:
                        sys_log.error("error in sync_check : {}".format(e))
                        continue

                    # ------------ enlist_rtc ------------ #
                    try:
                        #        ep_loc_pointcnt 에 따라 다르게 수행        #
                        if ep_loc_pointcnt == 1:
                            res_df = self.utils.enlist_rtc(res_df, config)
                        else:
                            res_df = self.utils1.enlist_rtc(res_df, config1)
                            res_df = self.utils2.enlist_rtc(res_df, config2)
                        sys_log.info('~ enlist_rtc time : %.5f' % (time.time() - temp_time))

                    except Exception as e:
                        sys_log.error("error in enlist_rtc : {}".format(e))
                        continue

                    # ------------ enlist_tr ------------ #
                    try:
                        np_timeidx = np.array(list(map(lambda x: intmin(x), res_df.index)))

                        #        1. ep_loc_pointcnt 에 따라 다르게 수행        #
                        #        2. ep 는 완전한 static, out 은 half-dynamic
                        #           a. initial_ep 를 저장해주어야함
                        if ep_loc_pointcnt == 1:
                            res_df = self.utils.enlist_tr(res_df, config, np_timeidx)
                        else:
                            res_df = self.utils1.enlist_tr(res_df, config1, np_timeidx)
                            res_df = self.utils2.enlist_tr(res_df, config2, np_timeidx)
                        sys_log.info('res_df.index[-1] : {}'.format(res_df.index[-1]))
                        sys_log.info('~ ep enlist time : %.5f\n' % (time.time() - temp_time))

                    except Exception as e:
                        sys_log.error('error in enlist_tr : {}'.format(e))
                        continue

                    # ---------------- ep_loc ---------------- #
                    if ep_loc_pointcnt == 0:
                        try:

                            #        save res_df before, entry const check      #
                            if config.trader_set.df_log:
                                excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                                # res_df.to_excel(df_log_path + "/%s.xlsx" % excel_name)
                                res_df.reset_index().to_feather(df_log_path + "/%s.ftr" % excel_name, compression='lz4')

                            #        1. config 에 매칭되게 entry 비교      #
                            #        2. 매칭된 strat. 로 config & utils 에 대입
                            # if res_df['entry_{}'.format(strat_version)][config.trader_set.last_index] == config.ep_set.short_entry_score:
                            if res_df['entry_{}'.format(config1.strat_version)][
                                config1.trader_set.last_index] == config1.ep_set.short_entry_score or \
                                    res_df['entry_{}'.format(config2.strat_version)][
                                        config2.trader_set.last_index] == config2.ep_set.short_entry_score:

                                if res_df['entry_{}'.format(config1.strat_version)][config1.trader_set.last_index] == \
                                        config1.ep_set.short_entry_score:
                                    config = config1
                                    self.utils = self.utils1
                                else:
                                    config = config2
                                    self.utils = self.utils2

                                initial_ep = res_df['short_ep_{}'.format(config.strat_version)].iloc[
                                    config.trader_set.last_index]

                                if not config.pos_set.short_ban:
                                    res_df, open_side, _ = self.utils_public.short_ep_loc(res_df, config,
                                                                                          config.trader_set.last_index)

                            # elif res_df['entry_{}'.format(strat_version)][config.trader_set.last_index] == -config.ep_set.short_entry_score:
                            elif res_df['entry_{}'.format(config1.strat_version)][
                                config1.trader_set.last_index] == -config1.ep_set.short_entry_score or \
                                    res_df['entry_{}'.format(config2.strat_version)][
                                        config2.trader_set.last_index] == -config2.ep_set.short_entry_score:

                                if res_df['entry_{}'.format(config1.strat_version)][config1.trader_set.last_index] == \
                                        -config1.ep_set.short_entry_score:
                                    config = config1
                                    self.utils = self.utils1
                                else:
                                    config = config2
                                    self.utils = self.utils2

                                print("ep_loc.point executed strat_ver : {}".format(config.strat_version))

                                initial_ep = res_df['long_ep_{}'.format(config.strat_version)].iloc[
                                    config.trader_set.last_index]

                                if not config.pos_set.long_ban:
                                    res_df, open_side, _ = self.utils_public.long_ep_loc(res_df, config,
                                                                                         config.trader_set.last_index)

                        except Exception as e:
                            sys_log.error("error in ep_loc phase : {}".format(e))
                            continue

                        if self.utils is not None:
                            ep_loc_pointcnt = 1
                            #      Todo     #
                            #        1. initial_ep 를 만들지말고, res_df_int
                            # print("ep_loc_pointcnt :", ep_loc_pointcnt)
                            # print("config.strat_version :", config.strat_version)
                            # print("open_side :", open_side)
                            # print("config.trader_set.symbol :", config.trader_set.symbol)
                            # quit()

                    ep_loc_check = 0  # once used, turn it off

                check_entry_sec = datetime.now().second
                if open_side is not None:

                    #        1. const. open market second
                    if config.ep_set.entry_type == "MARKET":
                        if check_entry_sec > config.trader_set.check_entry_sec:
                            open_side = None
                            sys_log.info("check_entry_sec : {}".format(check_entry_sec))
                            continue  # ep_loc_check = False 라서 open_side None phase 로 감

                    #        2. init fee        #
                    if config.ep_set.entry_type == "MARKET":
                        fee = config.trader_set.market_fee
                    else:
                        fee = config.trader_set.limit_fee

                    #        1. ep_loc.point2 - point1 과 동시성이 성립하지 않음, 가 위치할 곳      #
                    #           a. out 이 point 에 따라 변경되기 때문에 이곳이 적합한 것으로 판단     #
                    #        2. load_df phase loop 돌릴 것
                    #        3. continue 바로하면 안되고, 분마다 진행해야할 것 --> waiting_zone while loop 적용
                    #           a. 첫 point2 검사는 바로진행해도 될것
                    #           b. ei_k check 이 realtime 으로 진행되어야한다는 점
                    #               i. => ei_k check by df version

                    #        3. set strat_version        #
                    strat_version = config.strat_version

                    if strat_version == "v5_2":
                        if ep_loc_pointcnt == 1:
                            # print("v5_2 passed")
                            #        1. ei_k set       #
                            #        2. 위아래에 e_j 수정
                            #           a. e_j 에 관한 고찰 필요함, backtest 에는 i + 1 부터 시작하니 +1 하는게 맞을 것으로 봄
                            #       Todo        #
                            #        1. tp_j 도 고정 시켜야함 (현재 new res_df 에 의해 갱신되는 중)
                            tp_j = config.trader_set.last_index  # = initial_i
                            e_j = config.trader_set.last_index + 1

                            if open_side == OrderSide.SELL:
                                if res_df['low'].iloc[e_j] <= res_df['h_short_rtc_1_{}'.format(strat_version)].iloc[
                                    tp_j] - \
                                        res_df['h_short_rtc_gap_{}'.format(strat_version)].iloc[
                                            tp_j] * config.loc_set.zone.ei_k:
                                    sys_log.info("cancel open_order by ei_k\n")
                                    ep_out = 1

                                elif (res_df['dc_upper_1m'].iloc[e_j - 1] <= res_df['dc_upper_15m'].iloc[e_j]) & \
                                        (res_df['dc_upper_15m'].iloc[e_j - 1] != res_df['dc_upper_15m'].iloc[e_j]):
                                    break
                            else:
                                if res_df['high'].iloc[e_j] >= res_df['h_long_rtc_1_{}'.format(strat_version)].iloc[
                                    tp_j] + \
                                        res_df['h_long_rtc_gap_{}'.format(strat_version)].iloc[
                                            tp_j] * config.loc_set.zone.ei_k:
                                    sys_log.info("cancel open_order by ei_k\n")
                                    ep_out = 1

                                elif (res_df['dc_lower_1m'].iloc[e_j - 1] >= res_df['dc_lower_15m'].iloc[e_j]) & \
                                        (res_df['dc_lower_15m'].iloc[e_j - 1] != res_df['dc_lower_15m'].iloc[e_j]):
                                    break

                    else:
                        break

                    #   1. 추후 position change platform 으로 변경 가능
                    #    a. first_iter 사용

                # ------- open_side is None [ waiting zone ] ------- #

                #        1. ep_loc.point2 를 위해 이 phase 를 잘 활용해야할 것      #
                #        2. 아래 phase 를 while 문에 가둬놓으면 될 것 같은데
                # else:
                while 1:

                    #       check tick change      #
                    #       latest df confirmation       #
                    if datetime.timestamp(res_df.index[-1]) < datetime.now().timestamp():
                        ep_loc_check = 1
                        load_new_df = 1
                        sys_log.info('res_df[-1] timestamp : %s' % datetime.timestamp(res_df.index[-1]))
                        sys_log.info('current timestamp : %s' % datetime.now().timestamp() + "\n")

                        #       get configure every minutes      #
                        try:
                            with open(cfg_full_path1, 'r') as cfg:
                                config1 = EasyDict(json.load(cfg))
                            with open(cfg_full_path2, 'r') as cfg:
                                config2 = EasyDict(json.load(cfg))

                            config2.trader_set = config1.trader_set

                            if ep_loc_pointcnt == 0:    # config 가 수정되지 않은 경우만 config1 을 base 로 반영
                                config = config1

                        except Exception as e:
                            print("error in load config (waiting zone phase):", e)
                            time.sleep(1)
                            continue

                        try:
                            with open(dst_cfg_path, 'r') as sys_cfg:
                                sys_log_cfg = EasyDict(json.load(sys_cfg))
                                logging.config.dictConfig(sys_log_cfg)
                                sys_log = logging.getLogger()

                        except Exception as e:
                            print("error in load sys_log_cfg (waiting zone phase):", e)
                            time.sleep(config.trader_set.api_retry_term)
                            continue

                        break  # return to load_new_df

                    else:
                        time.sleep(config.trader_set.realtime_term)  # <-- term for realtime market function

                    # time.sleep(config.trader_set.close_complete_term)   # <-- term for close completion

                if ep_out:
                    break

            if ep_out:
                continue

            first_iter = True  # 포지션 변경하는 경우, 필요함
            while 1:  # <-- loop for 'check order type change condition'

                sys_log.info('open_side : {}'.format(open_side))

                # ------------- tr setting ------------- #
                # if config.ep_set.entry_type == OrderType.MARKET:
                #     #             ep by realtime_price            #
                #     try:
                #         ep = get_market_price_v2(self.sub_client)
                #     except Exception as e:
                #         sys_log.error('error in get_market_price : {}'.format(e))
                #         continue
                # else:

                #             limit & market's ep             #
                #             inversion adjusted
                if strat_version == "v5_2":
                    out_j = e_j
                    # ep_j = e_j
                else:
                    out_j = config.trader_set.last_index
                ep_j = config.trader_set.last_index
                tp_j = config.trader_set.last_index

                if open_side == OrderSide.BUY:
                    ep = res_df['long_ep_{}'.format(strat_version)].iloc[ep_j]

                    if config.pos_set.long_inversion:
                        open_side = OrderSide.SELL  # side change (inversion)
                        out = res_df['long_tp_{}'.format(strat_version)].iloc[out_j]
                        tp = res_df['long_out_{}'.format(strat_version)].iloc[tp_j]
                    else:
                        out = res_df['long_out_{}'.format(strat_version)].iloc[out_j]
                        tp = res_df['long_tp_{}'.format(strat_version)].iloc[tp_j]
                else:
                    ep = res_df['short_ep_{}'.format(strat_version)].iloc[ep_j]

                    if config.pos_set.short_inversion:
                        open_side = OrderSide.BUY
                        out = res_df['short_tp_{}'.format(strat_version)].iloc[out_j]
                        tp = res_df['short_out_{}'.format(strat_version)].iloc[tp_j]
                    else:
                        out = res_df['short_out_{}'.format(strat_version)].iloc[out_j]
                        tp = res_df['short_tp_{}'.format(strat_version)].iloc[tp_j]

                #         get price, volume precision --> reatlime 가격 변동으로 인한 precision 변동 가능성       #
                try:
                    price_precision, quantity_precision = get_precision(config.trader_set.symbol)
                except Exception as e:
                    sys_log.error('error in get price & volume precision : {}'.format(e))
                    time.sleep(config.trader_set.api_retry_term)
                    continue
                else:
                    sys_log.info('price_precision : {}'.format(price_precision))
                    sys_log.info('quantity_precision : {}'.format(quantity_precision))

                if strat_version == "v5_2":
                    ep = calc_with_precision(initial_ep, price_precision)
                else:  # --> e_j 사용하는 pr 이 더 높게 나옴
                    ep = calc_with_precision(ep, price_precision)

                #         open_price comparison       #
                open_price = res_df['open'].iloc[-1]
                if open_side == OrderSide.BUY:
                    ep = min(open_price, ep)
                else:
                    ep = max(open_price, ep)

                out = calc_with_precision(out, price_precision)  # includes half-dynamic out
                tp = calc_with_precision(tp, price_precision)  # includes half-dynamic tp
                leverage = self.utils_public.lvrg_set(res_df, config, open_side, ep, out, fee, limit_leverage)

                sys_log.info('ep : {}'.format(ep))
                sys_log.info('out : {}'.format(out))
                sys_log.info('tp : {}'.format(tp))
                sys_log.info('leverage : {}'.format(leverage))
                sys_log.info('~ ep out lvrg set time : %.5f' % (time.time() - temp_time))

                try:
                    request_client.change_initial_leverage(symbol=config.trader_set.symbol, leverage=leverage)
                except Exception as e:
                    sys_log.error('error in change_initial_leverage : {}'.format(e))
                    time.sleep(config.trader_set.api_retry_term)
                    continue  # -->  ep market 인 경우에 조심해야함 - why ..?
                else:
                    sys_log.info('leverage changed --> {}'.format(leverage))

                if first_iter:
                    # ---------- define start asset ---------- #
                    #       config1 의 setting 을 기준으로함       #
                    if self.accumulated_income == 0.0:

                        available_balance = config1.trader_set.initial_asset  # USDT

                    else:
                        if config1.trader_set.asset_changed:
                            available_balance = config1.trader_set.initial_asset

                            #       asset_changed -> 0      #
                            config1.trader_set.asset_changed = 0

                            with open(cfg_full_path1, 'w') as cfg:
                                json.dump(config1, cfg, indent=2)
                        else:
                            available_balance += income

                # ---------- get availableBalance ---------- #
                try:
                    max_available_balance = get_availableBalance()

                    #       over_balance 가 저장되어 있다면, 지속적으로 max_balance 와의 비교 진행      #
                    if self.over_balance is not None:
                        if self.over_balance <= max_available_balance * 0.9:
                            available_balance = self.over_balance
                            self.over_balance = None
                        else:
                            available_balance = max_available_balance * 0.9  # over_balance 를 넘지 않는 선에서 max_balance 채택

                    else:  # <-- 예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치
                        if available_balance > max_available_balance:
                            self.over_balance = available_balance
                            available_balance = max_available_balance * 0.9

                    if available_balance < self.min_balance:
                        sys_log.info('available_balance %.3f < min_balance\n' % available_balance)
                        break

                except Exception as e:
                    sys_log.error('error in get_availableBalance : {}\n'.format(e))
                    time.sleep(config.trader_set.api_retry_term)
                    continue

                sys_log.info('~ get balance time : %.5f' % (time.time() - temp_time))

                # ---------- get available open_quantity ---------- #
                open_quantity = available_balance / ep * leverage
                open_quantity = calc_with_precision(open_quantity, quantity_precision)
                if self.over_balance is not None:
                    sys_log.info('available_balance (temp) : {}'.format(available_balance))
                else:
                    sys_log.info('available_balance : {}'.format(available_balance))

                sys_log.info("open_quantity : {}".format(open_quantity))

                real_balance = ep * open_quantity
                sys_log.info("real_balance : {}".format(real_balance))

                # ---------- open order ---------- #
                orderside_changed = False
                order_info = (available_balance, leverage)
                self.over_balance, res_code = limit_order(self, config.ep_set.entry_type, config, open_side, ep,
                                                          open_quantity, order_info)

                if res_code:  # order deny except res_code = 0
                    break

                # -------- execution wait time -------- #
                if config.ep_set.entry_type == OrderType.MARKET:

                    #           market : prevent close at open bar           #
                    #           + enough time for open_quantity consumed
                    while 1:
                        if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                            break
                        else:
                            time.sleep(config.trader_set.realtime_term)  # <-- for realtime price function

                else:
                    #       limit's ep_out const. & check breakout_qty_ratio         #
                    first_exec_qty_check = 1
                    check_time = time.time()
                    while 1:

                        #       temporary - wait bar close       #
                        # if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):

                        #       check tp_done by hl with realtime_price     #
                        try:
                            realtime_price = get_market_price_v2(self.sub_client)

                        except Exception as e:
                            sys_log.error('error in get_market_price_v2 (ep_out phase) : {}'.format(e))
                            continue

                        #        1. tp_j 를 last_index 로 설정해도 갱신되지 않음 (res_df 가 갱신되지 않는한)
                        #       Todo        #
                        #        2. 추후, dynamic_tp 사용시 res_df 갱신해야할 것
                        #           a. 그에 따른 res_df 종속 변수 check
                        tp_j = config.trader_set.last_index

                        if open_side == OrderSide.SELL:
                            if realtime_price <= res_df['h_short_rtc_1_{}'.format(strat_version)].iloc[tp_j] - \
                                    res_df['h_short_rtc_gap_{}'.format(strat_version)].iloc[
                                        tp_j] * config.loc_set.zone.ei_k:
                                sys_log.info("cancel open_order by ei_k\n")
                                break
                        else:
                            if realtime_price >= res_df['h_long_rtc_1_{}'.format(strat_version)].iloc[tp_j] + \
                                    res_df['h_long_rtc_gap_{}'.format(strat_version)].iloc[
                                        tp_j] * config.loc_set.zone.ei_k:
                                sys_log.info("cancel open_order by ei_k\n")
                                break

                        time.sleep(config.trader_set.realtime_term)  # <-- for realtime price function

                        #        1. breakout_gty_ratio
                        #           a. exec_open_qty 와 open_quantity 의 ratio 비교
                        #           b. realtime_term 에 delay 주지 않기 위해, term 설정
                        #        2. 이걸 설정 안해주면 체결되도 close_order 진행 안됨
                        if first_exec_qty_check or time.time() - check_time >= config.trader_set.qty_check_term:

                            try:
                                exec_open_quantity = get_remaining_quantity(config.trader_set.symbol)
                            except Exception as e:
                                sys_log.error('error in exec_open_quantity check : {}'.format(e))
                                time.sleep(config.trader_set.api_retry_term)
                                continue
                            else:
                                first_exec_qty_check = 0
                                check_time = time.time()

                            if abs(exec_open_quantity / open_quantity) >= config.trader_set.breakout_qty_ratio:
                                break

                #           1. when, open order time expired or executed              #
                #           2. regardless to position exist, cancel all open orders       #
                while 1:
                    try:
                        request_client.cancel_all_orders(symbol=config.trader_set.symbol)
                    except Exception as e:
                        sys_log.error('error in cancel remaining open order : {}'.format(e))
                    else:
                        break

                if orderside_changed:
                    first_iter = False
                    sys_log.info('orderside_changed : {}\n'.format(orderside_changed))
                    continue
                else:
                    break

            #       reject unacceptable amount of asset     #
            if available_balance < self.min_balance:
                continue

            while 1:  # <-- for complete exec_open_quantity function
                try:
                    exec_open_quantity = get_remaining_quantity(config.trader_set.symbol)
                except Exception as e:
                    sys_log.error('error in exec_open_quantity check : {}'.format(e))
                    continue
                break

            if exec_open_quantity == 0.0:  # <-- open order 가 체결되어야 position 이 생김

                income = 0
                # continue

            #       position / 체결량 이 존재하면 close order 진행      #
            else:

                sys_log.info('open order executed')

                #        1. save trade_log      #
                #        2. check str(res_df.index[-2]), back_sync entry on close     #
                entry_timeindex = str(res_df.index[-2])
                trade_log[entry_timeindex] = [ep, open_side, "open"]

                with open(trade_log_fullpath, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    sys_log.info("entry trade_log dumped !\n")

                # ----------- close order ----------- #

                #           set close side          #
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                use_new_df2 = 0
                if not config.tp_set.static_tp or not config.out_set.static_out or strat_version == "v5_2":
                    use_new_df2 = 1
                load_new_df2 = 1
                limit_tp = 0
                remained_tp_id = None
                prev_remained_tp_id = None
                tp_exec_dict = {}
                cross_on = 0  # exist for early_out

                while 1:

                    #        1-1. load new_df in every new_df2 index change           #
                    #             -> 이전 index 와 비교하려면, 어차피 load_new_df 해야함      #

                    if use_new_df2:

                        if load_new_df2:  # dynamic_out & tp phase

                            try:
                                res_df_list = []
                                old_df = 0
                                for itv_i, interval_ in enumerate(config.trader_set.interval_list):

                                    if interval_ != "None":
                                        new_df_, _ = concat_candlestick(config.trader_set.symbol, interval_,
                                                                        days=1,
                                                                        limit=config.trader_set.row_list[itv_i],
                                                                        timesleep=0.2,
                                                                        show_process=False)
                                        if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                            old_df = 1
                                            break
                                        else:
                                            new_df = new_df_.iloc[-config.trader_set.row_list[itv_i]:].copy()
                                    else:
                                        new_df = None

                                    res_df_list.append(new_df)

                                if old_df:
                                    continue

                            except Exception as e:
                                sys_log.error('error in get new_dfs (load_new_df2 phase) : {}'.format(e))
                                continue

                            else:
                                load_new_df2 = 0

                            try:
                                res_df = self.utils_public.sync_check(res_df_list)
                                res_df = self.utils_public.public_indi(res_df)

                            except Exception as e:
                                sys_log.error("error in sync_check & public_indi (load_new_df2 phase) : {}".format(e))
                                continue

                            try:
                                res_df = self.utils.enlist_rtc(res_df, config)

                            except Exception as e:
                                sys_log.error('error in enlist_rtc (close phase) : {}'.format(e))
                                continue

                            try:
                                res_df = self.utils.enlist_tr(res_df, config, np_timeidx)
                                # sys_log.info("res_df.index[-1] (load_new_df2 phase) : {}".format(res_df.index[-1]))

                            except Exception as e:
                                sys_log.error('error in enlist_tr (close phase) : {}'.format(e))
                                continue

                            #        dynamic inversion platform adjusted         #
                            if open_side == OrderSide.BUY:
                                if config.pos_set.short_inversion:
                                    out_series = res_df['short_tp_{}'.format(strat_version)]
                                    tp_series = res_df['short_out_{}'.format(strat_version)]
                                else:
                                    out_series = res_df['long_out_{}'.format(strat_version)]
                                    tp_series = res_df['long_tp_{}'.format(strat_version)]
                            else:
                                if config.pos_set.long_inversion:
                                    out_series = res_df['long_tp_{}'.format(strat_version)]
                                    tp_series = res_df['long_out_{}'.format(strat_version)]
                                else:
                                    out_series = res_df['short_out_{}'.format(strat_version)]
                                    tp_series = res_df['short_tp_{}'.format(strat_version)]

                            if not config.out_set.static_out:
                                out = out_series.iloc[config.trader_set.last_index]

                            if not config.tp_set.static_tp:
                                tp = tp_series.iloc[config.trader_set.last_index]

                    # --------- limit tp order 여부 조사 phase --------- #
                    #          1. static_limit 은 reorder 할필요 없음
                    if config.tp_set.tp_type == OrderType.LIMIT or config.tp_set.tp_type == "BOTH":

                        prev_remained_tp_id = remained_tp_id

                        try:
                            remained_tp_id = remaining_order_check(config.trader_set.symbol)

                        except Exception as e:
                            sys_log.error('error in remaining_order_check : {}'.format(e))
                            time.sleep(config.trader_set.api_retry_term)
                            continue

                        if config.tp_set.static_tp:
                            if remained_tp_id is None:  # first tp_order
                                limit_tp = 1
                        else:
                            if remained_tp_id is None:  # first tp_order
                                limit_tp = 1
                            else:
                                #       tp 가 달라진 경우만 진행 = reorder     #
                                if tp_series.iloc[config.trader_set.last_index] != \
                                        tp_series.iloc[config.trader_set.last_index - 1]:
                                    limit_tp = 1

                                    #           미체결 tp order 모두 cancel               #
                                    #           regardless to position exist, cancel all tp orders       #
                                    while 1:
                                        try:
                                            request_client.cancel_all_orders(symbol=config.trader_set.symbol)
                                        except Exception as e:
                                            sys_log.error('error in cancel remaining tp order : {}'.format(e))
                                            time.sleep(config.trader_set.api_retry_term)
                                            continue
                                        else:
                                            break

                    # ------- limit_tp order phase ------- #
                    if limit_tp:

                        while 1:
                            #         get realtime price, volume precision       #
                            try:
                                price_precision, quantity_precision = get_precision(config.trader_set.symbol)
                            except Exception as e:
                                sys_log.error('error in get price & volume precision : {}'.format(e))
                                continue
                            else:
                                sys_log.info('price_precision : {}'.format(price_precision))
                                sys_log.info('quantity_precision : {}'.format(quantity_precision))
                                break

                        #        1. prev_remained_tp_id 생성
                        #       Todo        #
                        #        2. (partial) tp_list 를 직접 만드는 방법 밖에 없을까 --> 일단은.
                        #        3. tp_list 는 ultimate_tp 부터 정렬        #
                        # tp_list = [tp, tp2]
                        # tp_series_list = [tp_series, tp2_series]
                        tp_list = [tp]

                        if len(tp_list) > 1:  # if partial_tp,

                            #       Todo        #
                            #        0. partial_tp --> sub_tp 체결시마다 기록
                            #         a. 체결 confirm 은 len(remained_tp_id) 를 통해 진행
                            #        1. back_pr calculation
                            #        위해 체결된 시간 (res_df.index[-2]) 과 해당 tp 를 한번만 기록함        #

                            #            sub tp log phase            #
                            #       1. pop executed sub_tps from tp_list
                            #       2. log executed sub_tps
                            if remained_tp_id is not None and prev_remained_tp_id is not None:

                                #        sub_tp execution        #
                                if remained_tp_id != prev_remained_tp_id:
                                    #        log sub_tp        #
                                    executed_sub_tp = tp_series_list[-1].iloc[-2]
                                    executed_sub_prev_tp = tp_series_list[-1].iloc[-3]
                                    sys_log.info("executed_sub_prev_tp, executed_sub_tp : {}, {}"
                                                 .format(executed_sub_prev_tp, executed_sub_tp))
                                    tp_exec_dict[res_df.index[-2]] = [executed_sub_prev_tp, executed_sub_tp]

                                    #        pop sub_tp        #
                                    tp_list.pop()
                                    # tp_series_list.pop()
                                    sys_log.info("tp_list.pop() executed")

                        tp_list = list(map(lambda x: calc_with_precision(x, price_precision), tp_list))
                        sys_log.info('tp_list : {}'.format(tp_list))

                        while 1:
                            try:
                                result = partial_limit_v2(self, config, tp_list, close_side,
                                                          quantity_precision, config.tp_set.partial_qty_divider)

                                if result is not None:
                                    sys_log.info(result)
                                    time.sleep(config.trader_set.api_retry_term)
                                    continue

                            except Exception as e:
                                sys_log.error("error in partial_limit : {}".format(e))
                                time.sleep(config.trader_set.api_retry_term)
                                continue

                            else:
                                sys_log.info("limit tp order enlisted : {}".format(datetime.now()))
                                limit_tp = 0
                                break

                    # ---------- limit tp exec. check & market close survey ---------- #
                    #            1. limit close (tp) execution check, every minute                 #
                    #            2. check market close signal, simultaneously
                    limit_done = 0
                    market_close_on = 0
                    log_tp = np.nan
                    load_new_df3 = 1
                    while 1:

                        # --------- limit tp execution check phase --------- #
                        if config.tp_set.tp_type == OrderType.LIMIT or config.tp_set.tp_type == "BOTH":
                            try:
                                tp_remain_quantity = get_remaining_quantity(config.trader_set.symbol)
                            except Exception as e:
                                sys_log.error('error in get_remaining_quantity : {}'.format(e))
                                time.sleep(config.trader_set.api_retry_term)
                                continue

                            #       들고 있는 position quantity 가 0 이면, 이 거래를 끝임        #
                            #        1. if tp_remain_quantity < 1 / (10 ** quantity_precision): 로 변경되어야할 것
                            # if tp_remain_quantity == 0.0:
                            if tp_remain_quantity <= 1 / (10 ** quantity_precision):
                                # if tp_remain_quantity < 1 / (10 ** quantity_precision):
                                sys_log.info('all limit tp order executed')

                                #            1. market tp / out 에 대한 log 방식도 서술
                                #            2. dict 사용 이유 : partial_tp platform
                                if not config.tp_set.static_tp:
                                    tp_exec_dict[res_df.index[-2]] = [tp_series.iloc[config.trader_set.last_index - 1],
                                                                      tp]
                                else:
                                    tp_exec_dict[res_df.index[-2]] = [tp]
                                sys_log.info("tp_exec_dict : {}".format(tp_exec_dict))

                                limit_done = 1
                                break

                        # --------- market close signal check phase (out | tp) --------- #
                        try:

                            #        1. 1m df 는 분당 1번만 진행
                            #         a. 그렇지 않을 경우 realtime_price latency 에 영향 줌

                            #        back_pr wait_time 과 every minute end loop 의 new_df_ 를 위해 필요함
                            #        1.  load_new_df3 상시 개방해야하는 거 아닌가
                            if use_new_df2:
                                new_df_ = res_df
                            else:
                                # if config.tp_set.static_tp and config.out_set.static_out:
                                # if config.tp_set.tp_type == 'LIMIT' and config.out_set.hl_out:

                                try:
                                    if load_new_df3:
                                        new_df_, _ = concat_candlestick(config.trader_set.symbol,
                                                                        config.trader_set.interval_list[0],
                                                                        days=1, limit=config.trader_set.row_list[0],
                                                                        timesleep=0.2)
                                        if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                            continue

                                except Exception as e:
                                    sys_log.error('error in load_new_df3 : {}'.format(e))
                                    #        1. adequate term for retries
                                    time.sleep(config.trader_set.api_retry_term)
                                    continue

                                else:
                                    load_new_df3 = 0

                            #       exists for hl_out       #
                            if config.out_set.hl_out:
                                realtime_price = get_market_price_v2(self.sub_client)

                            # np_timeidx = np.array(list(map(lambda x: intmin(x), new_df_.index)))

                            # ---------- out & tp check ---------- #
                            if open_side == OrderSide.BUY:

                                #       long hl_out       #
                                if config.out_set.use_out:
                                    if config.out_set.hl_out:
                                        if realtime_price <= out:  # hl_out
                                            market_close_on = True
                                            log_tp = out
                                            sys_log.info("out : {}".format(out))

                                    #       long close_out       #
                                    else:
                                        j = config.trader_set.last_index
                                        if new_df_['close'].iloc[j] <= out:
                                            market_close_on = True
                                            log_tp = new_df_['close'].iloc[j]
                                            sys_log.info("out : {}".format(out))

                                #       long market tp        #
                                # if config.tp_set.tp_type == "MARKET":
                                if config.tp_set.tp_type == "MARKET" or config.tp_set.tp_type == "BOTH":

                                    #       early_out       #
                                    if strat_version == "v5_2":
                                        j = config.trader_set.last_index
                                        if res_df['close'].iloc[j] > res_df['bb_upper_5m'].iloc[j] > \
                                                res_df['close'].iloc[j - 1]:
                                            cross_on = 1
                                        if cross_on == 1 and res_df['close'].iloc[j] < res_df['bb_lower_5m'].iloc[j] < \
                                                res_df['close'].iloc[j - 1]:
                                            market_close_on = True
                                            log_tp = new_df_['close'].iloc[j]
                                            sys_log.info("tp : {}".format(tp))

                                    #       unknown strat.'s market tp logic       #
                                    # if new_df_['close'].iloc[config.trader_set.last_index] >= tp:
                                    #     market_close_on = True
                                    #     log_tp = new_df_['close'].iloc[config.trader_set.last_index]
                                    #     sys_log.info("tp : {}".format(tp))

                            else:
                                if config.out_set.use_out:
                                    #       short hl_out      #
                                    if config.out_set.hl_out:
                                        if realtime_price >= out:
                                            market_close_on = True
                                            log_tp = out
                                            sys_log.info("out : {}".format(out))

                                    #       short close_out     #
                                    else:
                                        j = config.trader_set.last_index
                                        if new_df_['close'].iloc[j] >= out:
                                            market_close_on = True
                                            log_tp = new_df_['close'].iloc[j]
                                            sys_log.info("out : {}".format(out))

                                #       short market tp      #
                                # if config.tp_set.tp_type == "MARKET":
                                if config.tp_set.tp_type == "MARKET" or config.tp_set.tp_type == "BOTH":

                                    if strat_version == "v5_2":
                                        j = config.trader_set.last_index
                                        if res_df['close'].iloc[j] < res_df['bb_lower_5m'].iloc[j] < \
                                                res_df['close'].iloc[j - 1]:
                                            cross_on = 1
                                        if cross_on == 1 and res_df['close'].iloc[j] > res_df['bb_upper_5m'].iloc[j] > \
                                                res_df['close'].iloc[j - 1]:
                                            market_close_on = True
                                            log_tp = new_df_['close'].iloc[j]
                                            sys_log.info("tp : {}".format(tp))

                                    # if new_df_['close'].iloc[config.trader_set.last_index] <= tp:
                                    #     market_close_on = True
                                    #     log_tp = new_df_['close'].iloc[config.trader_set.last_index]
                                    #     sys_log.info("tp : {}".format(tp))

                            if market_close_on:
                                if config.out_set.hl_out:
                                    sys_log.info("realtime_price : {}".format(realtime_price))
                                sys_log.info("market_close_on is True")

                                # --- log tp --- #
                                tp_exec_dict[new_df_.index[-2]] = [log_tp]
                                sys_log.info("tp_exec_dict : {}".format(tp_exec_dict))
                                break

                        except Exception as e:
                            sys_log.error('error in checking market close signal : {}'.format(e))
                            continue

                        # ------ minute end phase ------ #
                        if datetime.now().timestamp() > datetime.timestamp(new_df_.index[-1]):

                            if use_new_df2:
                                load_new_df2 = 1
                                break
                            else:
                                # if config.tp_set.tp_type == 'LIMIT' and config.out_set.hl_out:
                                #     pass
                                # else:
                                load_new_df3 = 1

                        else:
                            time.sleep(config.trader_set.realtime_term)

                    #   all limit close executed     #
                    if limit_done:
                        fee += config.trader_set.limit_fee
                        break

                    # if load_new_df2:    # new_df 와 dynamic limit_tp reorder 가 필요한 경우
                    #     continue

                    if not market_close_on:  # = load_new_df2
                        continue

                    #               market close phase - get market signal               #
                    else:  # market_close_on = 1

                        fee += config.trader_set.market_fee

                        remain_tp_canceled = False  # 미체결 limit tp 존재가능함
                        market_close_order(self, remain_tp_canceled, config, close_side, out, log_tp)

                        break  # <--- break for all close order loop, break partial tp loop

                # ----------------------- check back_pr ----------------------- #
                #       total_income() function confirming -> wait close confirm       #
                while 1:

                    lastest_close_timeindex = new_df_.index[-1]

                    if datetime.now().timestamp() > datetime.timestamp(lastest_close_timeindex):
                        break

                # ---------- back_pr calculation ---------- #
                #            1. adjust partial tp             #
                #            2. -1 이 아니라, timestamp index 로 접근해야할듯      #
                #            2-1. len(tp_list) 가 2 -> 1 로 변한 순간을 catch, time_idx 를 기록
                #            3. 마지막 체결 정보를 기록 : (res_df.index[-1]) 과 해당 tp 를 기록        #

                #             real_tp, division           #
                #             prev_tp == tp 인 경우는, tp > open 인 경우에도 real_tp 는 tp 와 동일함      #
                calc_tmp_profit = 1
                r_qty = 1  # base asset ratio
                # ---------- calc in for loop ---------- #
                for q_i, (k_ts, v_tp) in enumerate(sorted(tp_exec_dict.items(), key=lambda x: x[0], reverse=True)):

                    # ---------- get real_tp phase ---------- #
                    if len(v_tp) > 1:
                        prev_tp, tp = v_tp

                        if prev_tp == tp:
                            real_tp = tp

                        else:  # dynamic_tp 로 인한 open tp case
                            if close_side == OrderSide.BUY:
                                if tp > res_df['open'].loc[k_ts]:
                                    real_tp = res_df['open'].loc[k_ts]
                                    sys_log.info("market tp executed !")
                                else:
                                    real_tp = tp
                            else:
                                if tp < res_df['open'].loc[k_ts]:
                                    real_tp = res_df['open'].loc[k_ts]
                                    sys_log.info("market tp executed !")
                                else:
                                    real_tp = tp

                    else:  # market close's real_tp | out (partial_tp also have market out)
                        [real_tp] = v_tp

                    #       check partial_tp case        #
                    if len(tp_exec_dict) == 1:
                        temp_qty = r_qty
                    else:
                        # if q_i != 0:
                        if q_i != len(tp_exec_dict) - 1:
                            temp_qty = r_qty / config.tp_set.partial_qty_divider
                        else:
                            temp_qty = r_qty

                    r_qty -= temp_qty

                    if open_side == OrderSide.SELL:
                        calc_tmp_profit += (ep / real_tp - fee - 1) * temp_qty
                        # calc_tmp_profit += ep / real_tp - fee
                    else:
                        calc_tmp_profit += (real_tp / ep - fee - 1) * temp_qty

                    sys_log.info("real_tp : {}".format(real_tp))
                    sys_log.info("ep : {}".format(ep))
                    sys_log.info("fee : {}".format(fee))
                    sys_log.info("temp_qty : {}".format(temp_qty))

                    #            save exit data         #
                    #            exit_timeindex use "-1" index           #
                    # exit_timeindex = str(res_df.index[-1])
                    # trade_log[exit_timeindex] = [real_tp, "close"]
                    trade_log[k_ts] = [real_tp, "close"]

                with open(trade_log_fullpath, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    sys_log.info("exit trade_log dumped !")

                end_timestamp = int(time.time() * 1000)

                #           get total income from this trade       #
                while 1:
                    try:
                        income = total_income(config.trader_set.symbol, start_timestamp, end_timestamp)
                        self.accumulated_income += income
                        sys_log.info("income : {} USDT".format(income))

                    except Exception as e:
                        sys_log.error('error in total_income : {}'.format(e))

                    else:
                        break

                tmp_profit = income / real_balance * leverage
                self.accumulated_profit *= (1 + tmp_profit)
                self.calc_accumulated_profit *= 1 + (calc_tmp_profit - 1) * leverage
                sys_log.info(
                    'temporary profit : %.3f (%.3f) %%' % (tmp_profit * 100, (calc_tmp_profit - 1) * leverage * 100))
                sys_log.info('accumulated profit : %.3f (%.3f) %%' % (
                    (self.accumulated_profit - 1) * 100, (self.calc_accumulated_profit - 1) * 100))
                sys_log.info('accumulated income : {} USDT\n'.format(self.accumulated_income))
