import os

#        1. relative path should be static '/IDE' 를 가리켜야함, 기준은 script 실행 dir 기준 (bots, back_idep)
#        2. 깊이가 다르면, './../' 이런식의 표현으로는 동일 pkg_path 에 접근할 수 없음
# print(os.getcwd())
# pkg_path = os.path.abspath('./../')
pkg_path = r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\IDE"  # system env. 에 따라 가변적
os.chdir(pkg_path)

from funcs_binance.binance_futures_modules import *  # math, pandas, bot_config (API_key & clients)
# from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick
from funcs_binance.funcs_trader_modules import read_write_cfg_list, get_income_info, calc_ideal_profit, \
    get_new_df, check_out, check_market_tp, check_limit_tp_exec, log_sub_tp_exec, get_dynamic_tpout, check_breakout_qty, check_ei_k, get_balance, \
    get_eptpout, check_ei_k_onclose, init_set, get_open_side
from funcs_binance.funcs_order_logger_trailorder import limit_order, partial_limit_order_v2, cancel_order_list, market_close_order
from funcs.funcs_trader import intmin_np
import numpy as np  # for np.nan
import time
import pickle
import logging.config
from pathlib import Path
from easydict import EasyDict
from datetime import datetime
import shutil


class Trader:
    def __init__(self, utils_public, utils_list, config_name_list):
        # ------ ID info. ------ #
        self.utils_public = utils_public
        self.utils_list = utils_list
        self.config_name_list = config_name_list
        self.config_list = None
        self.config = None
        self.utils = None

        # ------ balance ------ #
        self.available_balance = None
        self.over_balance = None
        self.min_balance = 5.0  # USDT

        # ------ pnl ------ #
        self.income = 0.0
        self.accumulated_income = 0.0
        self.calc_accumulated_profit = 1.0
        self.accumulated_profit = 1.0
        self.accumulated_back_profit = 1.0

        self.sub_client = None

    def run(self):
        exec_file_name = Path(__file__).stem

        #       path definition     #
        log_path_list = ["sys_log", "trade_log", "df_log"]
        sys_log_path, trade_log_path, df_log_path = [os.path.join(pkg_path, path_, exec_file_name) for path_ in log_path_list]

        for path_ in [sys_log_path, trade_log_path, df_log_path]:
            os.makedirs(path_, exist_ok=True)

        cfg_path_list = [os.path.join(pkg_path, "config", name_) for name_ in self.config_name_list]

        initial_set = 1
        while 1:
            # ------ load config (refreshed by every trade) ------ #
            self.config_list = read_write_cfg_list(cfg_path_list)
            #       self.config = config1 & equal config.trader_set      #
            for cfg_idx, cfg_ in enumerate(self.config_list):
                if cfg_idx != 0:
                    cfg_.trader_set = self.config.trader_set
                else:
                    self.config = cfg_

            # ------ check symbol_changed ------ #
            if self.config.trader_set.symbol_changed:
                self.config_list[0].trader_set.symbol_changed = 0
                initial_set = 1  # symbol changed -> reset initial_set

            if initial_set:
                sys_log.info('# ----------- {} ----------- #'.format(exec_file_name))
                sys_log.info("pkg_path : {}".format(pkg_path))

                limit_leverage, self.sub_client = init_set(self)    # self.config 가 위에서 정의된 상황
                sys_log.info("initial_set done\n")

                # -------- rewrite modified self.config_list-------- #
                read_write_cfg_list(cfg_path_list, mode='w', edited_cfg_list=self.config_list)

                # -------- trade_log_fullpath define -------- #
                trade_log = {}
                trade_log_name = "{}_{}.pkl".format(self.config.trader_set.symbol,
                                                    str(datetime.now().timestamp()).split(".")[0])
                trade_log_fullpath = os.path.join(trade_log_path, trade_log_name)

                # -------- copy base_cfg.json -> {ticker}.json -------- #
                src_cfg_path = sys_log_path.replace(exec_file_name, "base_cfg.json")
                dst_cfg_path = os.path.join(sys_log_path, "{}.json".format(self.config.trader_set.symbol))
                # print("dst_cfg_path :", dst_cfg_path)
                # quit()

                try:
                    shutil.copy(src_cfg_path, dst_cfg_path)
                except Exception as e:
                    print("error in shutil.copy() :", e)
                    continue

                initial_set = 0

            # -------- set logger info. to {ticker}.json - offer realtime modification -------- #
            try:
                with open(dst_cfg_path, 'r') as sys_cfg:
                    sys_log_cfg = EasyDict(json.load(sys_cfg))

                if initial_set:  # dump edited_cfg
                    sys_log_cfg.handlers.file_rot.filename = \
                        os.path.join(sys_log_path, trade_log_name.replace(".pkl", ".log"))  # trade_log_name 에 종속
                    logging.getLogger("apscheduler.executors.default").propagate = False

                    with open(dst_cfg_path, 'w') as edited_cfg:
                        json.dump(sys_log_cfg, edited_cfg, indent=3)

                logging.config.dictConfig(sys_log_cfg)
                sys_log = logging.getLogger()

            except Exception as e:
                print("error in load sys_log_cfg :", e)
                time.sleep(self.config.trader_set.api_retry_term)
                continue

            # ------ check run ------ #
            if not self.config.trader_set.run:
                time.sleep(self.config.trader_set.realtime_term)  # give enough time to read config
                continue

            # ------ get start_timestamp ------ #
            start_timestamp = int(time.time() * 1000)

            # ------ param. init ------ #
            load_new_df = 1
            self.utils = None
            open_side = None
            ep_loc_point2 = 0
            ep_out = 0
            while 1:
                if load_new_df:
                    # ------ log last trading time ------ #
                    #        미체결을 고려해, load_new_df 마다 log 수행       #
                    trade_log["last_trading_time"] = str(datetime.now())
                    with open(trade_log_fullpath, "wb") as dict_f:
                        pickle.dump(trade_log, dict_f)

                    # ------------ get_new_df ------------ #
                    temp_time = time.time()
                    res_df, _, load_new_df = get_new_df(self)
                    sys_log.info("~ load_new_df time : %.2f" % (time.time() - temp_time))

                    # ------------ self.utils_ ------------ #
                    try:
                        res_df = self.utils_public.sync_check(res_df, self.config)  # function usage format maintenance
                        sys_log.info('~ sync_check time : %.5f' % (time.time() - temp_time))
                        np_timeidx = np.array([intmin_np(date_) for date_ in res_df.index.to_numpy()])
                        res_df = self.utils_public.public_indi(res_df, self.config, np_timeidx)
                        sys_log.info('~ public_indi time : %.5f' % (time.time() - temp_time))

                        #        ep_loc_point2 -> 해당 ID 로 수행        #
                        if ep_loc_point2:
                            res_df = self.utils.enlist_rtc(res_df, self.config, np_timeidx)
                            res_df = self.utils.enlist_tr(res_df, self.config, np_timeidx)
                        else:
                            for utils_, config_ in zip(self.utils_list, self.config_list):
                                res_df = utils_.enlist_rtc(res_df, config_, np_timeidx)
                                res_df = utils_.enlist_tr(res_df, config_, np_timeidx)
                        sys_log.info('~ enlist_rtc & enlist_tr time : %.5f' % (time.time() - temp_time))
                        sys_log.info('res_df.index[-1] : {}'.format(res_df.index[-1]))

                    except Exception as e:
                        sys_log.error("error in self.utils_ : {}".format(e))
                        continue

                    # ---------------- ep_loc - get_open_side ---------------- #
                    if open_side is None:   # open_signal not exists, check ep_loc
                        try:
                            if self.config.trader_set.df_log:   # save res_df at ep_loc
                                excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                                res_df.reset_index().to_feather(df_log_path + "/%s.ftr" % excel_name, compression='lz4')

                            open_side, self.utils, self.config = get_open_side(self, res_df, np_timeidx)

                        except Exception as e:
                            sys_log.error("error in ep_loc phase : {}".format(e))
                            continue

                        if open_side is not None:  # res_df_init 정의는 이곳에서 해야함 - 아래서 할 경우 res_df_init 갱신되는 문제
                            res_df_init = res_df.copy()

                # ------------ after load_new_df - check open_signal again ------------ #
                    # <-- 이줄에다가 else 로 아래 phase 옮길 수 없나 ? --> 안됨, ep_loc survey 진행후에도 아래 phase 는 실행되어야함
                check_entry_sec = datetime.now().second
                if open_side is not None:
                    # ------ check_entry_sec ------ #
                    if self.config.ep_set.entry_type == "MARKET":
                        if check_entry_sec > self.config.trader_set.check_entry_sec:
                            open_side = None
                            sys_log.info("check_entry_sec : {}".format(check_entry_sec))
                            continue  # ep_loc_check = False 라서 open_side None phase 로 감 -> 무슨 의미 ? 어쨌든 wait_zone 으로 회귀

                    # ------ init fee ------ #
                    if self.config.ep_set.entry_type == "MARKET":
                        fee = self.config.trader_set.market_fee
                    else:
                        fee = self.config.trader_set.limit_fee

                    #        1. ep_loc.point2 - point1 과 동시성이 성립하지 않음, 가 위치할 곳      #
                    #           a. out 이 point 에 따라 변경되기 때문에 이곳이 적합한 것으로 판단     #
                    #        2. load_df phase loop 돌릴 것
                    #        3. continue 바로하면 안되고, 분마다 진행해야할 것 --> waiting_zone while loop 적용
                    #           a. 첫 point2 검사는 바로진행해도 될것
                    #           b. ei_k check 이 realtime 으로 진행되어야한다는 점
                    #               i. => ei_k check by ID

                    # ------ set strat_version ------ #
                    strat_version = self.config.strat_version

                    # ------ point2 (+ ei_k) phase ------ #
                    if strat_version in ['v5_2']:
                        ep_loc_point2 = 1
                        sys_log.warning("v5_2 ep_loc_point2 = 1 passed")

                        #           a. e_j 에 관한 고찰 필요함, backtest 에는 i + 1 부터 시작하니 +1 하는게 맞을 것으로 봄
                        #               -> 좀더 명확하게, dc_lower_1m.iloc[i - 1] 에 last_index 가 할당되는게 맞아서
                        #        3. tp_j 도 고정 시켜야함 (현재 new res_df 에 의해 갱신되는 중)
                        tp_j = self.config.trader_set.complete_index  # = initial_i
                        e_j = self.config.trader_set.complete_index + 1

                        ep_out = check_ei_k_onclose(self, res_df_init, res_df, open_side, e_j, tp_j)
                        allow_ep_in, _ = self.utils_public.ep_loc_point2(res_df, self.config, e_j, out_j=None, allow_ep_in=0, side=open_side)
                        if allow_ep_in:
                            break
                    else:
                        break

                # -------------- open_side is None - no_signal holding zone -------------- #
                #        1. 추후 position change platform 으로 변경 가능
                #           a. first_iter 사용
                #        2. ep_loc.point2 를 위해 이 phase 를 잘 활용해야할 것      #
                while 1:
                    # ------- check bar_ends - latest df confirmation ------- #
                    if datetime.timestamp(res_df.index[-1]) < datetime.now().timestamp():
                        #       ep_loc.point2 를 위한 res_df 갱신이 필요하지 않은 경우,        #
                        #       ep_loc params. 초기화
                        if not ep_loc_point2:
                            self.utils = None
                        sys_log.info('res_df[-1] timestamp : %s' % datetime.timestamp(res_df.index[-1]))
                        sys_log.info('current timestamp : %s' % datetime.now().timestamp() + "\n")

                        # ------- get configure every bar_ends - trade_config ------- #
                        try:
                            self.config_list = read_write_cfg_list(cfg_path_list)
                        except Exception as e:
                            print("error in load config (waiting zone phase):", e)
                            time.sleep(1)
                            continue

                        # ------- sys_log configuration ------- #
                        try:
                            with open(dst_cfg_path, 'r') as sys_cfg:
                                sys_log_cfg = EasyDict(json.load(sys_cfg))
                                logging.config.dictConfig(sys_log_cfg)
                                sys_log = logging.getLogger()
                        except Exception as e:
                            print("error in load sys_log_cfg (waiting zone phase):", e)
                            time.sleep(self.config.trader_set.api_retry_term)
                            continue

                        load_new_df = 1
                        break  # return to load_new_df

                    else:
                        time.sleep(self.config.trader_set.realtime_term)  # <-- term for realtime market function
                        # time.sleep(self.config.trader_set.close_complete_term)   # <-- term for close completion

                if ep_out:  # 1m_bar close 까지 기다리는 logic
                    break

            if ep_out:  # ep_out init 을 위한 continue
                continue

            first_iter = True  # 포지션 변경하는 경우, 필요함
            while 1:  # <-- loop for 'check order type change condition'
                ep, tp, out, open_side = get_eptpout(self, e_j, open_side, res_df_init, res_df)

                # ------ get realtime price, volume precision - precision 변동 가능성 ------ #
                try:
                    price_precision, quantity_precision = get_precision(self.config.trader_set.symbol)
                except Exception as e:
                    sys_log.error('error in get price & volume precision : {}'.format(e))
                    time.sleep(self.config.trader_set.api_retry_term)
                    continue
                else:
                    sys_log.info('price_precision : {}'.format(price_precision))
                    sys_log.info('quantity_precision : {}'.format(quantity_precision))

                ep, tp, out = [calc_with_precision(price_, price_precision) for price_ in [ep, tp, out]]    #  includes half-dynamic tp

                # ------ open_price comparison ------ #
                open_price = res_df['open'].iloc[-1]
                if open_side == OrderSide.BUY:
                    ep = min(open_price, ep)
                else:
                    ep = max(open_price, ep)

                leverage = self.utils_public.lvrg_set(res_df, self.config, open_side, ep, out, fee, limit_leverage)

                sys_log.info('ep : {}'.format(ep))
                sys_log.info('out : {}'.format(out))
                sys_log.info('tp : {}'.format(tp))
                sys_log.info('leverage : {}'.format(leverage))
                sys_log.info('~ ep out lvrg set time : %.5f' % (time.time() - temp_time))

                while 1:
                    try:
                        request_client.change_initial_leverage(symbol=self.config.trader_set.symbol, leverage=leverage)
                    except Exception as e:
                        sys_log.error('error in change_initial_leverage : {}'.format(e))
                        time.sleep(self.config.trader_set.api_retry_term)
                        continue  # -->  ep market 인 경우에 조심해야함 - why ..?
                    else:
                        sys_log.info('leverage changed --> {}'.format(leverage))
                        break

                self.available_balance, self.over_balance, min_bal_bool = get_balance(self, first_iter, cfg_path_list)
                if min_bal_bool:
                    break
                sys_log.info('~ get balance time : %.5f' % (time.time() - temp_time))

                # ---------- calc. open_quantity ---------- #
                open_quantity = self.available_balance / ep * leverage
                open_quantity = calc_with_precision(open_quantity, quantity_precision)
                sys_log.info("open_quantity : {}".format(open_quantity))

                real_balance = ep * open_quantity
                sys_log.info("real_balance : {}".format(real_balance))  # define for pnl calc.

                # ---------- open order ---------- #
                orderside_changed = False
                order_info = (self.available_balance, leverage)
                post_order_res, self.over_balance, res_code = limit_order(self, self.config.ep_set.entry_type, self.config, open_side, ep,
                                                                          open_quantity, order_info)
                if res_code:  # order deny exception, res_code = 0
                    break

                # ------------ execution wait time ------------ #
                # ------ market : prevent close at open bar ------ #
                if self.config.ep_set.entry_type == OrderType.MARKET:
                    #           + enough time for open_quantity consumed
                    while 1:
                        if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                            break
                        else:
                            time.sleep(self.config.trader_set.realtime_term)  # <-- for realtime price function
                # ------ limit : check ep_out (= ei_k & breakout_qty) ------ #
                else:
                    first_exec_qty_check = 1
                    check_time = time.time()
                    while 1:
                        if check_ei_k(self, res_df_init, open_side):
                            break
                        first_exec_qty_check, check_time, breakout = check_breakout_qty(self, first_exec_qty_check,
                                                                                        check_time, post_order_res, open_quantity)
                        if breakout:
                            break

                # ------ when, open order time expired or executed ------ #
                #        regardless to position exist, cancel open orders       #
                open_executedQty = cancel_order_list(self.config.trader_set.symbol, [post_order_res], self.config.ep_set.entry_type)

                if orderside_changed:  # future_module
                    first_iter = False
                    sys_log.info('orderside_changed : {}\n'.format(orderside_changed))
                    continue
                else:
                    break

            # ------ reject unacceptable amount of asset ------ #
            if min_bal_bool:
                continue

            if open_executedQty == 0.0:  # open_executedQty 는 분명 정의됨
                self.income = 0
            else:
                sys_log.info('open order executed')

                #        1. save trade_log      #
                #        2. check str(res_df.index[-2]), back_sync entry on close     #
                #        Todo       #
                #         3. back_pr 에서 res_df.index[-2] line 에 무엇을 기입하냐에 따라 달라짐
                init_entry_timeindex = str(res_df_init.index[-2])
                entry_timeindex = str(res_df.index[-2])
                trade_log[init_entry_timeindex] = [open_side, "init_open"]
                trade_log[entry_timeindex] = [ep, open_side, "open"]

                with open(trade_log_fullpath, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    sys_log.info("entry trade_log dumped !\n")

                # ------ set close side ------ #
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                #   a. 아래의 조건문을 담을 변수가 필요함 - 병합 불가 (latest)
                #       i. => load_new_df2 는 1 -> 0 으로 변경됨
                use_new_df2 = 0
                if not self.config.tp_set.static_tp or not self.config.out_set.static_out or \
                        strat_version in ['v5_2', 'v7_3'] or self.config.tp_set.time_tp:
                    use_new_df2 = 1  # level close_tp / out 이 dynamic 한 경우 사용

                load_new_df2 = 1
                limit_tp = 0
                post_order_res_list = []
                tp_executedQty = 0
                tp_exec_dict = {}
                cross_on = 0  # exist for early_out
                while 1:
                    #     Todo : load new_df in every new_df2 index change <-- ?     #
                    if use_new_df2:
                        if load_new_df2:  # dynamic_out & tp phase
                            res_df, _, load_new_df2 = get_new_df(self, mode="CLOSE")
                            try:
                                res_df = self.utils_public.sync_check(res_df, self.config, order_side="CLOSE")
                                np_timeidx = np.array([intmin_np(date_) for date_ in res_df.index.to_numpy()])  # should be locate af. row_slice
                                res_df = self.utils_public.public_indi(res_df, self.config, np_timeidx, order_side="CLOSE")
                                res_df = self.utils.enlist_rtc(res_df, self.config, np_timeidx)
                                res_df = self.utils.enlist_tr(res_df, self.config, np_timeidx, mode="CLOSE")

                            except Exception as e:
                                sys_log.error("error in utils_ (load_new_df2 phase) : {}".format(e))
                                continue

                            tp, out, tp_series, out_series = get_dynamic_tpout(self, res_df, open_side, tp, out)

                    # --------- limit_tp check for order --------- #
                    if self.config.tp_set.tp_type == OrderType.LIMIT or self.config.tp_set.tp_type == "BOTH":
                        if len(post_order_res_list) == 0:  # 아무 limit_tp 진행하지 않은 상태
                            limit_tp = 1
                        else:
                            # ------ dynamic_tp reorder ------ #
                            if tp_series.iloc[self.config.trader_set.complete_index] != \
                                    tp_series.iloc[self.config.trader_set.complete_index - 1]:
                                limit_tp = 1
                                #        1. 본래는, remaining 으로 open_executedQty refresh 진행함,
                                #        2. 지금은, 직접 구해야할 것
                                #           a. refreshedQty = open_executedQty - tp_executedQty : error 발생할때마다 recalc.
                                #        3. cancel post_order_res in list
                                tp_executedQty = cancel_order_list(self.config.trader_set.symbol, post_order_res_list)

                    # ------- limit_tp order ------- #
                    if limit_tp:
                        # ------ get realtime price, qty precision ------ #
                        while 1:
                            try:
                                price_precision, quantity_precision = get_precision(self.config.trader_set.symbol)
                            except Exception as e:
                                sys_log.error('error in get price & volume precision : {}'.format(e))
                                continue
                            else:
                                sys_log.info('price_precision : {}'.format(price_precision))
                                sys_log.info('quantity_precision : {}'.format(quantity_precision))
                                break

                        #       Todo        #
                        #        2. (partial) tp_list 를 직접 만드는 방법 밖에 없을까 --> 일단은.
                        #        3. tp_list 는 ultimate_tp 부터 정렬        #
                        # tp_list = [tp, tp2]
                        # tp_series_list = [tp_series, tp2_series]  # for dynamic_tp
                        tp_list = [tp]
                        try:
                            tp_list, tp_exec_dict = log_sub_tp_exec(self, res_df, tp_list, post_order_res_list, quantity_precision, tp_series_list, tp_exec_dict)
                        except:
                            pass
                        tp_list = list(map(lambda x: calc_with_precision(x, price_precision), tp_list))
                        sys_log.info('tp_list : {}'.format(tp_list))

                        while 1:
                            try:
                                post_order_res_list = partial_limit_order_v2(self, self.config, tp_list, close_side,
                                                                             open_executedQty - tp_executedQty, quantity_precision,
                                                                             self.config.tp_set.partial_qty_divider)
                            except Exception as e:
                                sys_log.error("error in partial_limit_order_v2 : {}".format(e))
                                time.sleep(self.config.trader_set.api_retry_term)
                                #   Todo - tp_executedQty miscalc.
                                #       1. -2019 : Margin is insufficient 으로 나타날 것으로 보임
                                continue
                            else:
                                sys_log.info("limit tp order enlisted : {}".format(datetime.now()))
                                limit_tp = 0
                                break

                    # ------------ limit_tp exec. & market_close check ------------ #
                    #            1. limit close (tp) execution check, every minute                 #
                    #            2. check market close signal, simultaneously
                    limit_done = 0
                    market_close_on = 0
                    log_tp = None
                    load_new_df3 = 1
                    while 1:
                        # ------ limit_tp execution check ------ #
                        if check_limit_tp_exec(self, post_order_res_list, quantity_precision):
                            if not self.config.tp_set.static_tp:
                                tp_exec_dict[res_df.index[-2]] = [tp_series.iloc[self.config.trader_set.complete_index - 1], tp]
                                # Todo - dynamic_tp 인 경우에 대한 'tp' issue
                            else:
                                tp_exec_dict[res_df.index[-2]] = [tp]
                            sys_log.info("tp_exec_dict : {}".format(tp_exec_dict))
                            limit_done = 1
                            break

                        # ------ market_close check (= out & market_tp) ------ #
                        try:
                            #        1. 1m df 는 분당 1번만 진행
                            #           a. 그렇지 않을 경우 realtime_price latency 에 영향 줌
                            #        2. back_pr wait_time 과 every minute end loop 의 new_df_ 를 위해 필요함
                            if not use_new_df2:  # if use_new_df2, res_df = new_df_
                                if load_new_df3:
                                    res_df, _, load_new_df3 = get_new_df(self, calc_rows=False, mode="CLOSE")

                            market_close_on, log_tp = check_out(self, res_df, open_side, out)
                            market_close_on, log_tp, cross_on = check_market_tp(self, res_df, open_side, cross_on, tp)
                            if market_close_on:
                                sys_log.info("market_close_on is True")
                                # ------ log tp ------ #
                                tp_exec_dict[res_df.index[-2]] = [log_tp]  # market_close_on = True, log_tp != None (None 도 logging 가능하긴함)
                                sys_log.info("tp_exec_dict : {}".format(tp_exec_dict))
                                break

                        except Exception as e:
                            sys_log.error('error in checking market_close_on : {}'.format(e))
                            continue

                        # ------ bar_end phase - loop selection ------ #
                        if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                            if use_new_df2:
                                load_new_df2 = 1  # return to outer loop - get df2's data
                                break
                            else:
                                load_new_df3 = 1  # return to current loop
                        else:
                            time.sleep(self.config.trader_set.realtime_term)

                    # ------------ limit / market execution or load_new_df2 ------------ #
                    if limit_done:
                        fee += self.config.trader_set.limit_fee
                        break
                    if not market_close_on:  # = load_new_df2
                        continue
                    else:
                        # ------ market close phase ------ #
                        fee += self.config.trader_set.market_fee
                        remain_tp_canceled = False  # 미체결 limit tp 존재가능함
                        market_close_order(self, self.config, remain_tp_canceled, post_order_res_list, close_side, open_executedQty, out, log_tp)
                        # market 시 out var. 미사용
                        break  # <--- break for all close order loop, break partial tp loop

                # ------ total_income() function confirming -> wait close confirm ------ #
                while 1:
                    latest_close_timeidx = res_df.index[-1]
                    if datetime.now().timestamp() > datetime.timestamp(latest_close_timeidx):
                        break

                # ------------ calc_ideal_profit ------------ #
                ideal_profit, trade_log = calc_ideal_profit(self, open_side, close_side, tp_exec_dict, res_df, ep, fee, trade_log)

                with open(trade_log_fullpath, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    sys_log.info("exit trade_log dumped !")

                # ------------ get total income from this trade ------------ #
                self.income, self.accumulated_income, self.accumulated_profit, self.calc_accumulated_profit = \
                    get_income_info(self, self.config.trader_set.symbol, real_balance, leverage, ideal_profit, start_timestamp,
                                    int(time.time() * 1000))
