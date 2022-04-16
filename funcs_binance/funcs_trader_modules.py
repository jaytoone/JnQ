from funcs_binance.binance_futures_modules import total_income, OrderSide, OrderType, get_market_price_v2, \
    get_order_info, check_exec_by_order_info, get_availableBalance, request_client, sub_client, FuturesMarginType, get_limit_leverage, \
    callback, error, check_execution, get_execPrice, get_execQty, calc_with_precision
from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick
from funcs.funcs_trader import calc_rows_and_days
import numpy as np
import pandas as pd
from datetime import datetime
from easydict import EasyDict
import json
import time
import logging
from ast import literal_eval

sys_log = logging.getLogger()


def read_write_cfg_list(cfg_path_list, mode='r', edited_cfg_list=None):
    try:
        cfg_file_list = [open(cfg_path, mode) for cfg_path in cfg_path_list]
        if mode == 'r':
            cfg_list = [EasyDict(json.load(cfg_)) for cfg_ in cfg_file_list]
        elif mode == 'w':
            assert edited_cfg_list is not None, "assert edited_cfg_list is not None"
            _ = [json.dump(cfg_, cfg_file_, indent=2) for cfg_, cfg_file_ in zip(edited_cfg_list, cfg_file_list)]
        else:
            assert mode in ['r', 'w'], "assert mode in ['r', 'w']"

        #       opened files should be closed --> 닫지 않으면 reopen 시 error occurs         #
        _ = [cfg_.close() for cfg_ in cfg_file_list]

        if mode == 'r':
            return cfg_list
        else:
            return
    except Exception as e:
        sys_log.error("error in read_write_cfg_list :", e)


def init_set(self):
    while 1:  # for completion
        # ------ 0. position mode => hedge_mode (long & short) ------ #
        try:
            request_client.change_position_mode(dualSidePosition=True)
        except Exception as e:
            sys_log.error('error in change_position_mode : {}'.format(e))
        else:
            sys_log.info('position mode --> hedge_mode')

        # ------ 1. leverage type => "isolated" ------ #
        try:
            request_client.change_margin_type(symbol=self.config.trader_set.symbol,
                                              marginType=FuturesMarginType.ISOLATED)
        except Exception as e:
            sys_log.error('error in change_margin_type : {}'.format(e))
        else:
            sys_log.info('leverage type --> isolated')

        # ------ 2. confirm limit leverage ------ #
        try:
            limit_leverage = get_limit_leverage(symbol_=self.config.trader_set.symbol)
        except Exception as e:
            sys_log.error('error in get limit_leverage : {}'.format(e))
            continue
        else:
            sys_log.info('limit_leverage : {}'.format(limit_leverage))

        # ------ 3. sub_client ------ #
        try:
            sub_client.subscribe_aggregate_trade_event(self.config.trader_set.symbol.lower(), callback, error)
        except Exception as e:
            sys_log.error('error in get sub_client : {}'.format(e))
            continue
        else:
            return limit_leverage, sub_client

def get_open_side_v2(self, res_df, np_timeidx):
    open_side = None
    for utils_, cfg_ in zip(self.utils_list, self.config_list):
        # ------ 1. point ------ #
        #  여기 왜 latest_index 사용했는지 모르겠음 -> bar_close point 를 고려해 mr_res 와의 index 분리
        #   vecto. 하는 이상, point x ep_loc 의 index 는 sync. 되어야함 => complete_index 사용 (solved)
        #  Todo, 각 phase 분리한 이유 = sys_logging (solved)
        if res_df['short_open_{}'.format(cfg_.strat_version)].to_numpy()[cfg_.trader_set.complete_index]:
            sys_log.warning("[ short ] true_point strat_version : {}".format(cfg_.strat_version))
            # ------ 2. ban ------ #
            if not cfg_.pos_set.short_ban:
                # ------ 3. mr_res ------ #
                mr_res, zone_arr = self.utils_public.ep_loc_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL)
                sys_log.warning("mr_res[cfg_.trader_set.complete_index] : {}\n".format(mr_res[cfg_.trader_set.complete_index]))
                # ------ assign ------ #
                if mr_res[cfg_.trader_set.complete_index]:
                    self.utils = utils_
                    self.config = cfg_
                    open_side = OrderSide.SELL
                    break
            else:
                sys_log.warning("cfg_.pos_set.short_ban : {}".format(cfg_.pos_set.short_ban))

        # ------ 1. point ------ #
        #       'if' for nested ep_loc.point        #
        if res_df['long_open_{}'.format(cfg_.strat_version)].to_numpy()[cfg_.trader_set.complete_index]:
            sys_log.warning("[ long ] true_point strat_version : {}".format(cfg_.strat_version))
            # ------ 2. ban ------ #
            if not cfg_.pos_set.long_ban:  # additional const. on trader
                # ------ 3. mr_res ------ #
                mr_res, zone_arr = self.utils_public.ep_loc_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY)
                sys_log.warning("mr_res[cfg_.trader_set.complete_index] : {}\n".format(mr_res[cfg_.trader_set.complete_index]))
                # ------ assign ------ #
                if mr_res[cfg_.trader_set.complete_index]:
                    self.utils = utils_
                    self.config = cfg_
                    open_side = OrderSide.BUY
                    break
            else:
                sys_log.warning("cfg_.pos_set.long_ban : {}".format(cfg_.pos_set.long_ban))

    return open_side, self.utils, self.config

def get_open_side(self, res_df, np_timeidx):
    #        Todo : list comprehension 가능할 것        #
    open_side = None
    for utils_, cfg_ in zip(self.utils_list, self.config_list):
        #       open_score     #
        if res_df['short_open_{}'.format(cfg_.strat_version)][cfg_.trader_set.latest_index] == cfg_.ep_set.short_entry_score:
            sys_log.warning("[ short ] ep_loc.point executed strat_ver : {}".format(cfg_.strat_version))
            #       ban      #
            if not cfg_.pos_set.short_ban:
                #       ep_loc      #
                res_df, open_side, _ = self.utils_public.short_ep_loc(res_df, cfg_,
                                                                cfg_.trader_set.complete_index,
                                                                np_timeidx)
                sys_log.warning("open_side : {}".format(open_side))
                #       assign      #
                if open_side is not None:
                    self.utils = utils_
                    self.config = cfg_
                    break
            else:
                sys_log.warning("cfg_.pos_set.short_ban : {}".format(cfg_.pos_set.short_ban))

        #       open_score     #
        #       'if' for nested ep_loc.point        #
        if res_df['long_open_{}'.format(cfg_.strat_version)][cfg_.trader_set.latest_index] == \
                -cfg_.ep_set.short_entry_score:
            sys_log.warning("[ long ] ep_loc.point executed strat_ver : {}".format(cfg_.strat_version))
            #       ban      #
            if not cfg_.pos_set.long_ban:  # additional const. on trader
                #       ep_loc      #
                res_df, open_side, _ = self.utils_public.long_ep_loc(res_df, cfg_,
                                                                cfg_.trader_set.complete_index,
                                                                np_timeidx)
                sys_log.warning("open_side : {}".format(open_side))
                #       assign      #
                if open_side is not None:
                    self.utils = utils_
                    self.config = cfg_
                    break
            else:
                sys_log.warning("cfg_.pos_set.long_ban : {}".format(cfg_.pos_set.long_ban))

    return open_side, self.utils, self.config

def get_streamer(self):
    row_list = literal_eval(self.config.trader_set.row_list)
    itv_list = literal_eval(self.config.trader_set.itv_list)
    rec_row_list = literal_eval(self.config.trader_set.rec_row_list)
    use_rows, days = calc_rows_and_days(itv_list, row_list, rec_row_list)

    back_df = pd.read_feather(self.config.trader_set.back_data_path, columns=None, use_threads=True).set_index("index")

    if self.config.trader_set.start_datetime != "None":
        start_idx = np.argwhere(back_df.index == pd.to_datetime(self.config.trader_set.start_datetime)).item()
    else:
        start_idx = use_rows

    assert start_idx >= use_rows, "more dataframe rows required"

    for i in range(start_idx + 1, len(back_df)):    # +1 for i inclusion
        yield back_df.iloc[i - use_rows:i]

def get_new_df_onstream(self):
    try:
        res_df = next(self.streamer)  # Todo, 무결성 검증 미진행
    except Exception as e:
        sys_log.error("error in get_new_df_onstream : {}".format(e))
        quit()   # error 처리할 필요없이, backtrader 프로그램 종료
        # return None, None   # output len 유지
    else:
        return res_df, 0


def get_new_df(self, calc_rows=True, mode="OPEN"):
    while 1:
        try:
            row_list = literal_eval(self.config.trader_set.row_list)
            if calc_rows:
                itv_list = literal_eval(self.config.trader_set.itv_list)
                rec_row_list = literal_eval(self.config.trader_set.rec_row_list)
                use_rows, days = calc_rows_and_days(itv_list, row_list, rec_row_list)
            else:
                use_rows, days = row_list[0], 1   # load_new_df3 사용 의도 = close, time logging
            new_df_, _ = concat_candlestick(self.config.trader_set.symbol, '1m',
                                            days=days,
                                            limit=use_rows,
                                            timesleep=None,
                                            show_process=0)
            if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():  # ts validation 1
                time.sleep(self.config.trader_set.realtime_term)
                continue
            if mode == "OPEN":
                sys_log.info("complete new_df_'s time : {}".format(datetime.now()))

        except Exception as e:
            sys_log.error("error in get_new_df : {}".format(e))
            #        1. adequate term for retries
            time.sleep(self.config.trader_set.api_retry_term)
            continue
        else:
            return new_df_, 0    # 0 for load_new_df var.

def get_balance(self, first_iter, cfg_path_list):
    min_bal = 0
    if first_iter:
        # ---------- define initial asset ---------- #
        #       self.config 의 setting 을 기준으로함       #
        if self.accumulated_income == 0.0:
            self.available_balance = self.config.trader_set.initial_asset  # USDT
        else:
            self.available_balance += self.income

        # ---------- asset_change - 첫 거래여부와 무관하게 진행 ---------- #
        if self.config.trader_set.asset_changed:
            self.available_balance = self.config.trader_set.initial_asset
            #        1. 이런식으로, cfg 의 값을 변경(dump)하는 경우를 조심해야함 - self.config_list[0] 에 입력해야한다는 의미
            self.config_list[0].trader_set.asset_changed = 0
            with open(cfg_path_list[0], 'w') as cfg:
                json.dump(self.config_list[0], cfg, indent=2)

            sys_log.info("asset_changed 1 --> 0")

    # ---------- get availableBalance ---------- #
    if not self.config.trader_set.backtrade:
        max_available_balance = get_availableBalance()

        #       over_balance 가 저장되어 있다면, 지속적으로 max_balance 와의 비교 진행      #
        if self.over_balance is not None:
            if self.over_balance <= max_available_balance * 0.9:
                self.available_balance = self.over_balance
                self.over_balance = None
            else:
                self.available_balance = max_available_balance * 0.9  # over_balance 를 넘지 않는 선에서 max_balance 채택
            sys_log.info('available_balance (temp) : {}'.format(self.available_balance))

        else:  # <-- 예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치
            if self.available_balance > max_available_balance:
                self.over_balance = self.available_balance
                self.available_balance = max_available_balance * 0.9
            sys_log.info('available_balance : {}'.format(self.available_balance))

        if self.available_balance < self.min_balance:
            sys_log.info('available_balance {:.3f} < min_balance\n'.format(self.available_balance))
            min_bal = 1

    return self.available_balance, self.over_balance, min_bal


def get_eptpout(self, e_j, open_side, res_df_open, res_df):
    sys_log.info('open_side : {}'.format(open_side))
    strat_version = self.config.strat_version
    if strat_version in ['v5_2']:
        out_j = e_j
        # ep_j = e_j
    else:
        out_j = self.config.trader_set.complete_index
    ep_j = self.config.trader_set.complete_index
    tp_j = self.config.trader_set.complete_index

    #        1. 추후 other strat. 적용시 res_df_open recheck
    #       Todo        #
    #           b. --> res_df 사용시, dynamic (pre, whole) 을 허용하겠다는 의미
    #               i. ep_loc_point2 를 사용하지 않는 이상 res_df = res_df_open
    if open_side == OrderSide.BUY:
        ep = res_df_open['long_ep_{}'.format(strat_version)].iloc[ep_j]

        if self.config.pos_set.long_inversion:
            open_side = OrderSide.SELL  # side change (inversion)
            out = res_df['long_tp_{}'.format(strat_version)].iloc[out_j]
            tp = res_df_open['long_out_{}'.format(strat_version)].iloc[tp_j]
        else:
            out = res_df['long_out_{}'.format(strat_version)].iloc[out_j]
            tp = res_df_open['long_tp_{}'.format(strat_version)].iloc[tp_j]
    else:
        ep = res_df_open['short_ep_{}'.format(strat_version)].iloc[ep_j]

        if self.config.pos_set.short_inversion:
            open_side = OrderSide.BUY
            out = res_df['short_tp_{}'.format(strat_version)].iloc[out_j]
            tp = res_df_open['short_out_{}'.format(strat_version)].iloc[tp_j]
        else:
            out = res_df['short_out_{}'.format(strat_version)].iloc[out_j]
            tp = res_df_open['short_tp_{}'.format(strat_version)].iloc[tp_j]

    return ep, tp, out, open_side

def get_tpepout(self, open_side, res_df_open, res_df):
    sys_log.info('open_side : {}'.format(open_side))
    strat_version = self.config.strat_version

    tp_j = self.config.trader_set.complete_index
    ep_j = self.config.trader_set.complete_index
    # Todo, complete_index + 1 이 아니라, 해당 index 롤 res_df_open 이 아닌 -> res_df 에 입력하면 되는거아닌가
    #  -> use_point2 여부에 따라 다르게 작성할 필요가 없어짐 (solved)
    out_j = self.config.trader_set.complete_index

    # if strat_version in ['v5_2']:
    # if self.config.ep_set.point2.use_point2:
    #     # out_j = self.config.trader_set.complete_index + 1  # Todo, why + 1 ?
    #     out_j = self.config.trader_set.complete_index
    # else:
    #     out_j = self.config.trader_set.complete_index

    #       Todo        #
    #        b. --> res_df 사용시, dynamic (pre, whole) 을 허용하겠다는 의미
    #           i. ep_loc_point2 미사용시, res_df = res_df_open
    if open_side == OrderSide.BUY:
        ep = res_df_open['long_ep_{}'.format(strat_version)].to_numpy()[ep_j]

        if self.config.pos_set.long_inversion:
            open_side = OrderSide.SELL  # side change (inversion)
            out = res_df['long_tp_{}'.format(strat_version)].to_numpy()[out_j]
            tp = res_df_open['long_out_{}'.format(strat_version)].to_numpy()[tp_j]
        else:
            out = res_df['long_out_{}'.format(strat_version)].to_numpy()[out_j]
            tp = res_df_open['long_tp_{}'.format(strat_version)].to_numpy()[tp_j]
    else:
        ep = res_df_open['short_ep_{}'.format(strat_version)].to_numpy()[ep_j]

        if self.config.pos_set.short_inversion:
            open_side = OrderSide.BUY
            out = res_df['short_tp_{}'.format(strat_version)].to_numpy()[out_j]
            tp = res_df_open['short_out_{}'.format(strat_version)].to_numpy()[tp_j]
        else:
            out = res_df['short_out_{}'.format(strat_version)].to_numpy()[out_j]
            tp = res_df_open['short_tp_{}'.format(strat_version)].to_numpy()[tp_j]

    return tp, ep, out, open_side

def check_breakout_qty(self, first_exec_qty_check, check_time, post_order_res, open_quantity):
    #        1. breakout_gty_ratio
    #           a. exec_open_qty 와 open_quantity 의 ratio 비교
    #           b. realtime_term 에 delay 주지 않기 위해, term 설정
    #        2. 이걸 설정 안해주면 체결되도 close_order 진행 안됨
    if first_exec_qty_check or time.time() - check_time >= self.config.trader_set.qty_check_term:
        order_info = get_order_info(self.config.trader_set.symbol, post_order_res.orderId)
        breakout = abs(get_execQty(order_info) / open_quantity) >= self.config.trader_set.breakout_qty_ratio
        return 0, time.time(), breakout
    return 0, check_time, 0  # breakout_qty survey 실행한 경우만 check_time 갱신


def check_ei_k_v2(self, res_df_open, res_df, open_side):
    ep_out = 0
    strat_version = self.config.strat_version

    if self.config.loc_set.zone.ep_out_tick != "None":
        if time.time() - datetime.timestamp(res_df_open.index[self.config.trader_set.latest_index]) \
                >= self.config.loc_set.zone.ep_out_tick * 60:
            ep_out = 1

    if self.config.loc_set.zone.ei_k != "None":  # Todo -> 이거 None 이면 "무한 대기" 발생 가능함
        realtime_price = get_market_price_v2(self.sub_client)
        tp_j = self.config.trader_set.complete_index
        #       Todo        #
        #        2. 추후, dynamic_tp 사용시 res_df 갱신해야할 것
        #           a. 그에 따른 res_df 종속 변수 check
        #        3. ei_k - ep_out 변수 달아주고, close bar waiting 추가할지 고민중
        #        4. warning - 체결량 존재하는데, ep_out 가능함 - market 고민중
        #        5. funcs_trader_modules 에서 order_side check 하는 function 모두 inversion 고려해야할 것
        #           a. "단, order_side change 후로만 해당됨
        if open_side == OrderSide.SELL:
            short_tp_ = res_df_open['short_tp_{}'.format(strat_version)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
            short_tp_gap_ = res_df_open['short_tp_gap_{}'.format(strat_version)].to_numpy()
            if realtime_price <= short_tp_[tp_j] + short_tp_gap_[tp_j] * self.config.loc_set.zone.ei_k:
                ep_out = 1
        else:
            long_tp_ = res_df_open['long_tp_{}'.format(strat_version)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
            long_tp_gap_ = res_df_open['long_tp_gap_{}'.format(strat_version)].to_numpy()
            if realtime_price >= long_tp_[tp_j] - long_tp_gap_[tp_j] * self.config.loc_set.zone.ei_k:
                ep_out = 1
        time.sleep(self.config.trader_set.realtime_term)  # <-- for realtime price function

    if ep_out:
        sys_log.warning("cancel open_order by ei_k\n")

    return ep_out

def check_ei_k(self, res_df_open, open_side):
    ep_out = 0
    strat_version = self.config.strat_version
    if self.config.loc_set.zone.ei_k != "None":  # Todo -> 이거 None 이면 "무한 대기" 발생 가능함
        #       check tp_done by hl with realtime_price     #
        realtime_price = get_market_price_v2(self.sub_client)
        #        1. tp_j 를 last_index 로 설정해도 갱신되지 않음 (res_df 가 갱신되지 않는한)
        #        3. ei_k 사용 여부를 결정할 수 있도록, strat_version 에 따라서도 상이해질 것
        #       Todo        #
        #        2. 추후, dynamic_tp 사용시 res_df 갱신해야할 것
        #           a. 그에 따른 res_df 종속 변수 check
        #        3. ei_k - ep_out 변수 달아주고, close bar waiting 추가할지 고민중
        #        4. warning - 체결량 존재하는데, ep_out 가능함 - market 고민중
        #        5. funcs_trader_modules 에서 order_side check 하는 function 모두 inversion 고려해야할 것
        #           a. "단, order_side change 후로만 해당됨
        if self.config.tp_set.time_tp:
            # open bar_close 로부터의 elapsed time
            if time.time() - datetime.timestamp(res_df_open.index[-1]) >= self.config.loc_set.point.tf_entry * 30:    # half_time (seconds)
                ep_out = 1
        else:
            tp_j = self.config.trader_set.complete_index
            if open_side == OrderSide.SELL:
                if realtime_price <= res_df_open['h_short_rtc_1_{}'.format(strat_version)].iloc[tp_j] - \
                        res_df_open['h_short_rtc_gap_{}'.format(strat_version)].iloc[
                            tp_j] * self.config.loc_set.zone.ei_k:
                    ep_out = 1
            else:
                if realtime_price >= res_df_open['h_long_rtc_1_{}'.format(strat_version)].iloc[tp_j] + \
                        res_df_open['h_long_rtc_gap_{}'.format(strat_version)].iloc[
                            tp_j] * self.config.loc_set.zone.ei_k:
                    ep_out = 1
        time.sleep(self.config.trader_set.realtime_term)  # <-- for realtime price function

    if ep_out:
        sys_log.warning("cancel open_order by ei_k\n")

    return ep_out

def check_ei_k_onbarclose_v2(self, res_df_open, res_df, e_j, tp_j, open_side):     # for point2
    ep_out = 0
    strat_version = self.config.strat_version

    if self.config.loc_set.zone.ep_out_tick != "None":
        if datetime.timestamp(res_df.index[-1]) - datetime.timestamp(res_df_open.index[self.config.trader_set.latest_index]) \
                >= self.config.loc_set.zone.ep_out_tick * 60:
            ep_out = 1

    if self.config.loc_set.zone.ei_k != "None":  # Todo - onbarclose 에서는, res_df_open 으로 open_index 의 tp 정보를 사용
        if open_side == OrderSide.SELL:
            low = res_df['low'].to_numpy()
            short_tp_ = res_df_open['short_tp_{}'.format(strat_version)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
            short_tp_gap_ = res_df_open['short_tp_gap_{}'.format(strat_version)].to_numpy()
            if low[e_j] <= short_tp_[tp_j] + short_tp_gap_[tp_j] * self.config.loc_set.zone.ei_k:
                ep_out = 1
        else:
            high = res_df['high'].to_numpy()
            long_tp_ = res_df_open['long_tp_{}'.format(strat_version)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
            long_tp_gap_ = res_df_open['long_tp_gap_{}'.format(strat_version)].to_numpy()
            if high[e_j] >= long_tp_[tp_j] - long_tp_gap_[tp_j] * self.config.loc_set.zone.ei_k:
                ep_out = 1

    if ep_out:
        sys_log.warning("cancel open_order by ei_k\n")

    return ep_out

def check_ei_k_onbarclose(self, res_df_open, res_df, open_side, e_j, tp_j):     # for point2
    ep_out = 0
    strat_version = self.config.strat_version
    if open_side == OrderSide.SELL:
        if res_df['low'].iloc[e_j] <= res_df_open['h_short_rtc_1_{}'.format(strat_version)].iloc[tp_j] - \
                res_df_open['h_short_rtc_gap_{}'.format(strat_version)].iloc[tp_j] * self.config.loc_set.zone.ei_k:
            ep_out = 1
    else:
        if res_df['high'].iloc[e_j] >= res_df_open['h_long_rtc_1_{}'.format(strat_version)].iloc[tp_j] + \
                res_df_open['h_long_rtc_gap_{}'.format(strat_version)].iloc[tp_j] * self.config.loc_set.zone.ei_k:
            ep_out = 1

    if ep_out:
        sys_log.warning("cancel open_order by ei_k\n")

    return ep_out

def get_dynamic_tpout(self, res_df, open_side, tp, out):
    # ------ dynamic & inversion tp_out ------ #
    strat_version = self.config.strat_version
    if open_side == OrderSide.BUY:
        if self.config.pos_set.short_inversion:
            out_series = res_df['short_tp_{}'.format(strat_version)]
            tp_series = res_df['short_out_{}'.format(strat_version)]
        else:
            out_series = res_df['long_out_{}'.format(strat_version)]
            tp_series = res_df['long_tp_{}'.format(strat_version)]
    else:
        if self.config.pos_set.long_inversion:
            out_series = res_df['long_tp_{}'.format(strat_version)]
            tp_series = res_df['long_out_{}'.format(strat_version)]
        else:
            out_series = res_df['short_out_{}'.format(strat_version)]
            tp_series = res_df['short_tp_{}'.format(strat_version)]

    if not self.config.tp_set.static_tp:
        tp = tp_series.to_numpy()[self.config.trader_set.complete_index]
    if not self.config.out_set.static_out:
        out = out_series.to_numpy()[self.config.trader_set.complete_index]

    return tp, out, tp_series, out_series


def get_p_tpqty(self, ep, tp, open_executedQty, price_precision, quantity_precision, close_side):
    p_ranges, p_qty_ratio = literal_eval(self.config.tp_set.p_ranges), literal_eval(self.config.tp_set.p_qty_ratio)
    len_p = len(p_ranges)
    assert len_p == len(p_qty_ratio)
    en_ps, tps, open_qtys = [np.tile(arr_, (len_p,)) for arr_ in [ep, tp, open_executedQty]]

    if close_side == OrderSide.BUY:  # short
        p_tps = en_ps - (en_ps - tps) * p_ranges
    else:  # long
        p_tps = en_ps + (tps - en_ps) * p_ranges
    p_qtys = open_qtys * p_qty_ratio

    p_tps = list(map(lambda x: calc_with_precision(x, price_precision), p_tps))
    p_qtys = list(map(lambda x: calc_with_precision(x, quantity_precision), p_qtys))
    sys_log.info('p_tps : {}'.format(p_tps))
    sys_log.info('p_qtys : {}'.format(p_qtys))

    return p_tps, p_qtys


def check_limit_tp_exec_v2(self, post_order_res_list, quantity_precision, return_price=False):
    all_executed = 0
    order_info_list = [get_order_info(self.config.trader_set.symbol, post_order_res.orderId)
                       for post_order_res in post_order_res_list if post_order_res is not None]
    order_exec_check_list = [check_execution(order_info, quantity_precision) for order_info in order_info_list]

    if np.sum(order_exec_check_list) == len(order_exec_check_list):
        sys_log.info('all limit tp order executed')
        all_executed = 1

    if return_price:  # order_list 로 변환하는김에 같이 넣은 거임
        execPrice_list = [get_execPrice(order_info) for order_info, executed in zip(order_info_list, order_exec_check_list) if executed]
        return all_executed, execPrice_list

# def check_limit_tp_exec(self, post_order_res_list, quantity_precision, return_price=False):
#     all_executed = 0
#     if self.config.tp_set.tp_type == OrderType.LIMIT or self.config.tp_set.tp_type == "BOTH":
#         # order_exec_check_list = [check_exec_by_order_info(self.config.trader_set.symbol, post_order_res, quantity_precision, return_price, order_type)
#         #                          for post_order_res in post_order_res_list if post_order_res is not None]
#         order_info_list = [get_order_info(self.config.trader_set.symbol, post_order_res.orderId)
#                            for post_order_res in post_order_res_list if post_order_res is not None]
#         order_exec_check_list = [check_execution(order_info, quantity_precision) for order_info in order_info_list]
#
#         if np.sum(order_exec_check_list) == len(order_exec_check_list):
#             sys_log.info('all limit tp order executed')
#             all_executed = 1
#
#         if return_price:  # order_list 로 변환하는김에 같이 넣은 거임
#             execPrice_list = [get_execPrice(order_info) for order_info, executed in zip(order_info_list, order_exec_check_list) if executed]
#             return all_executed, execPrice_list
#
#     return all_executed, []  # => 체결된 tp 가 없음, return_length 는 유지해야할 것


# old
def log_sub_tp_exec(self, res_df, tp_list, post_order_res_list, quantity_precision, tp_series_list, tp_exec_dict):
    if len(tp_list) > 1:  # allow only partial_tp
        #        1. back_pr calculation 위해 체결된 시간 (res_df.index[-2]) 과 해당 tp 를 한번만 기록함        #
        #       Todo        #
        #        1. partial_tp 시, tp_series_list recheck

        # ------ sub tp execution logging ------ #
        #       1. pop executed sub_tps from tp_list
        #       2. log executed sub_tps
        #       Todo        #
        #        1. sub_tp execution 을 remained_tp_id 사용하지 않고 어떤식으로 가능할지
        #           a. post_order_res_list 의 order_res -> get_order_info() 진행, execution 확인
        #               i. exec_check logic
        order_exec_check_list = [check_exec_by_order_info(self.config.trader_set.symbol, post_order_res, quantity_precision)
                                 for post_order_res in post_order_res_list if post_order_res is not None]
        if np.sum(order_exec_check_list) != 0:  # = limit_tp execution exist
            executed_sub_tp = tp_series_list[-1].iloc[-2]  # complete_idx
            executed_sub_prev_tp = tp_series_list[-1].iloc[-3]
            sys_log.info("executed_sub_prev_tp, executed_sub_tp : {}, {}"
                         .format(executed_sub_prev_tp, executed_sub_tp))
            tp_exec_dict[res_df.index[-2]] = [executed_sub_prev_tp, executed_sub_tp]

            # ------ pop sub_tp ------ #
            tp_list.pop()   # this return popped value
            # tp_series_list.pop()
            sys_log.info("tp_list.pop() executed")

    return tp_list, tp_exec_dict

def check_hl_out_onbarclose(self, res_df, market_close_on, log_out, out, open_side):
    #    Todo - inversion 고려 아직임
    close = res_df['close'].to_numpy()
    c_i = self.config.trader_set.complete_index
    # ------------ open_side 기준 ------------ #
    if open_side == OrderSide.SELL:
        # ------------ out ------------ #
        # ------ short - hl_out ------ #
        if self.config.out_set.hl_out:
            high = res_df['high'].to_numpy()[c_i]
            const_str = "high >= out"
            if eval(const_str):
                market_close_on = True
                log_out = out
                sys_log.info("{} : {} {}".format(const_str, high, out))

        # ------ short - close_out ------ #
        else:
            if close[c_i] >= out:
                market_close_on = True
                log_out = close[c_i]
                sys_log.info("{} : {} {}".format("close >= out", log_out, out))
    else:
        # ------------ out ------------ #
        # ------ long - hl_out ------ #
        if self.config.out_set.hl_out:
            low = res_df['low'].to_numpy()[c_i]
            const_str = "low <= out"
            if eval(const_str):
                market_close_on = True
                log_out = out
                sys_log.info("{} : {} {}".format(const_str, low, out))

        # ------ long - close_out ------ #
        else:
            if close[c_i] <= out:
                market_close_on = True
                log_out = close[c_i]
                sys_log.info("{} : {} {}".format("close <= out", log_out, out))

    return market_close_on, log_out

def check_hl_out(self, res_df, market_close_on, log_out, out, open_side):
    #    Todo - inversion 고려 아직임
    close = res_df['close'].to_numpy()
    # ------------ open_side 기준 ------------ #
    if open_side == OrderSide.SELL:
        # ------------ out ------------ #
        # ------ short - hl_out ------ #
        if self.config.out_set.hl_out:
            realtime_price = get_market_price_v2(self.sub_client)
            const_str = "realtime_price >= out"
            if eval(const_str):
                market_close_on = True
                log_out = out
                sys_log.info("{} : {} {}".format(const_str, realtime_price, out))

        # ------ short - close_out ------ #
        else:
            j = self.config.trader_set.complete_index
            if close[j] >= out:
                market_close_on = True
                log_out = close[j]
                sys_log.info("{} : {} {}".format("close >= out", log_out, out))
    else:
        # ------------ out ------------ #
        # ------ long - hl_out ------ #
        if self.config.out_set.hl_out:
            realtime_price = get_market_price_v2(self.sub_client)
            const_str = "realtime_price <= out"
            if eval(const_str):
                market_close_on = True
                log_out = out
                sys_log.info("{} : {} {}".format(const_str, realtime_price, out))

        # ------ long - close_out ------ #
        else:
            j = self.config.trader_set.complete_index
            if close[j] <= out:
                market_close_on = True
                log_out = close[j]
                sys_log.info("{} : {} {}".format("close <= out", log_out, out))

    return market_close_on, log_out

def check_signal_out(self, res_df, market_close_on, log_out, cross_on, open_side):
    strat_version = self.config.strat_version

    #   Todo, inversion 에 대한 고려 진행된건가 - 안된것으로 보임
    close = res_df['close'].to_numpy()
    if open_side == OrderSide.SELL:
        # ------------ market_close ------------ #
        # ------ short ------ #
        j = self.config.trader_set.complete_index
        # ------ 1. rsi_exit ------ #
        if self.config.out_set.rsi_exit:
            rsi_ = res_df['rsi_%s' % self.config.loc_set.point.exp_itv].to_numpy()
            osc_band = self.config.loc_set.point.osc_band
            if (rsi_[j - 1] >= 50 - osc_band) & (rsi_[j] < 50 - osc_band):
                market_close_on = True

        # ------ 2. early_out ------ #
        if strat_version in ['v5_2']:
            bb_lower_5m = res_df['bb_lower_5m'].to_numpy()
            bb_upper_5m = res_df['bb_upper_5m'].to_numpy()
            if close[j] < bb_lower_5m[j] < close[j - 1]:
                cross_on = 1
            if cross_on == 1 and close[j] > bb_upper_5m[j] > close[j - 1]:
                market_close_on = True

        if market_close_on:
            log_out = close[j]
            sys_log.info("signal out : {}".format(log_out))
    else:
        # ------ long ------ #
        j = self.config.trader_set.complete_index
        # ------ 1. rsi_exit ------ #
        if self.config.out_set.rsi_exit:
            rsi_ = res_df['rsi_%s' % self.config.loc_set.point.exp_itv].to_numpy()
            osc_band = self.config.loc_set.point.osc_band
            if (rsi_[j - 1] <= 50 + osc_band) & (rsi_[j] > 50 + osc_band):
                market_close_on = True

        # ------ 2. early_out ------ #
        if strat_version in ['v5_2']:
            bb_lower_5m = res_df['bb_lower_5m'].to_numpy()
            bb_upper_5m = res_df['bb_upper_5m'].to_numpy()
            if close[j] > bb_upper_5m[j] > close[j - 1]:
                cross_on = 1
            if cross_on == 1 and close[j] < bb_lower_5m[j] < close[j - 1]:
                market_close_on = True

        if market_close_on:
            log_out = close[j]    # j 때문에 이곳에 배치
            sys_log.info("signal out : {}".format(log_out))

    return market_close_on, log_out, cross_on

# def get_income_info(self, real_balance, leverage, ideal_profit, real_profit, start_timestamp, end_timestamp):
#     while 1:
#         try:
#             income = total_income(self.config.trader_set.symbol, start_timestamp, end_timestamp)
#             self.accumulated_income += income
#             sys_log.info("income : {} USDT".format(income))
#         except Exception as e:
#             sys_log.error("error in total_income : {}".format(e))
#         else:
#             break
#
#     tmp_profit = income / real_balance * leverage
#     real_tmp_profit = (real_profit - 1) * leverage
#     ideal_tmp_profit = (ideal_profit - 1) * leverage
#
#     self.accumulated_profit *= 1 + tmp_profit
#     self.ideal_accumulated_profit *= 1 + ideal_tmp_profit
#
#     sys_log.info('calc_income : {} USDT'.format(real_balance * (real_profit - 1)))
#     sys_log.info('temporary profit : {:.2%} {:.2%} ({:.2%})'.format(tmp_profit, real_tmp_profit, ideal_tmp_profit))
#     sys_log.info('accumulated profit : {:.2%} ({:.2%})'.format((self.accumulated_profit - 1), (self.ideal_accumulated_profit - 1)))
#     sys_log.info('accumulated income : {} USDT\n'.format(self.accumulated_income))
#
#     return income, self.accumulated_income, self.accumulated_profit, self.ideal_accumulated_profit


def get_income_info_v2(self, real_balance, leverage, ideal_profit, real_profit):  # 복수 pos 를 허용하기 위한 api 미사용
    real_profit_pct = real_profit - 1
    income = real_balance * real_profit_pct
    self.accumulated_income += income

    sys_log.info('real_profit : {}'.format(real_profit))
    sys_log.info('ideal_profit : {}'.format(ideal_profit))

    real_tmp_profit = real_profit_pct * leverage
    ideal_tmp_profit = (ideal_profit - 1) * leverage

    self.accumulated_profit *= 1 + real_tmp_profit
    self.ideal_accumulated_profit *= 1 + ideal_tmp_profit

    sys_log.info('income : {} USDT'.format(income))
    sys_log.info('temporary profit : {:.2%} ({:.2%})'.format(real_tmp_profit, ideal_tmp_profit))
    sys_log.info('accumulated profit : {:.2%} ({:.2%})'.format((self.accumulated_profit - 1), (self.ideal_accumulated_profit - 1)))
    sys_log.info('accumulated income : {} USDT\n'.format(self.accumulated_income))

    return income, self.accumulated_income, self.accumulated_profit, self.ideal_accumulated_profit


# def calc_ideal_profit(self, open_side, close_side, tp_exec_dict, res_df, ep, fee, trade_log):
#     #   Todo - 추후에 list_comprehension 진행
#     #    1. real_tp_list 구하기
#     #    2. temp_qty_list 와 matching, order_side 기준으로 np.sum(calc_pr_list) 구하기
#     #    3. open_side & close_side 변수 합
#     ideal_profit = 1
#     r_qty = 1  # base asset ratio
#     # ------------ calc in for loop ------------ #
#     for q_i, (k_ts, v_tp) in enumerate(sorted(tp_exec_dict.items(), key=lambda x: x[0], reverse=True)):  # ts 기준 정렬
#         # ------ get real_tp ------ #
#         if len(v_tp) > 1:
#             prev_tp, tp = v_tp
#             if prev_tp == tp:
#                 real_tp = tp
#             else:  # dynamic_tp 로 인한 open tp case
#                 if close_side == OrderSide.BUY:
#                     if tp > res_df['open'].loc[k_ts]:
#                         real_tp = res_df['open'].loc[k_ts]
#                         sys_log.info("market tp executed !")
#                     else:
#                         real_tp = tp
#                 else:
#                     if tp < res_df['open'].loc[k_ts]:
#                         real_tp = res_df['open'].loc[k_ts]
#                         sys_log.info("market tp executed !")
#                     else:
#                         real_tp = tp
#         else:  # market close's real_tp or out (partial_tp also have market out)
#             [real_tp] = v_tp
#
#         # ------ check partial_tp case ------ #
#         if len(tp_exec_dict) == 1:
#             temp_qty = r_qty
#         else:
#             # if q_i != 0:
#             if q_i != len(tp_exec_dict) - 1:
#                 temp_qty = r_qty / self.config.tp_set.partial_qty_divider
#             else:
#                 temp_qty = r_qty
#
#         r_qty -= temp_qty
#
#         if open_side == OrderSide.SELL:
#             ideal_profit += (ep / real_tp - fee - 1) * temp_qty
#             # ideal_profit += ep / real_tp - fee
#         else:
#             ideal_profit += (real_tp / ep - fee - 1) * temp_qty
#
#         sys_log.info("real_tp : {}".format(real_tp))
#         sys_log.info("ep : {}".format(ep))
#         sys_log.info("fee : {}".format(fee))
#         sys_log.info("temp_qty : {}".format(temp_qty))
#
#         # ------------ save exit data ------------ #
#         #       exit_timeindex use "-1" index        #
#         # exit_timeindex = str(res_df.index[-1])
#         # trade_log[exit_timeindex] = [real_tp, "close"]
#         trade_log[k_ts] = [real_tp, "close"]
#
#     return ideal_profit, trade_log

# deprecated
def calc_ideal_profit_v4(self, res_df, open_side, ideal_ep, tp_exec_dict, open_executedPrice_list,
                         tp_executedPrice_list, out_executedPrice_list, p_qtys, fee):
    # 1. ideal_ep & tp_exec_dict exist for ideal_pr
    # 2. open_executedPrice_list, close_executedPrice_list exist for real_pr
    #   1. ideal
    #       a. (tp_exec_dict / ideal_ep - fee - 1) * p_qtys (long)
    #           i. dict item -> np.array
    #   2. real
    #   3. open_comparison 생략
    ideal_exp_np = np.hstack(np.array(list(tp_exec_dict.values())))  # list 로 저장된 value, 1d_array 로 conversion
    ideal_enp_np = np.array([ideal_ep] * len(ideal_exp_np))
    # market_close 에서 -2002, -4003 발생한 경우 = error 인데 close 된 경우
    # -> ideal_tp 적용
    if out_executedPrice_list is None:
        out_executedPrice_list = list(ideal_exp_np[-1:])    # list type 사용해야 append 구현됨
    exp_np = np.array(tp_executedPrice_list + out_executedPrice_list)
    enp_np = np.array(open_executedPrice_list * len(exp_np))

    p_qtys_np = np.array(p_qtys)
    if open_side == OrderSide.SELL:
        ideal_profit = ((ideal_enp_np / ideal_exp_np - fee - 1) * p_qtys_np).sum(axis=1) + 1
        real_profit = ((enp_np / exp_np - fee - 1) * p_qtys_np).sum(axis=1) + 1
    else:
        ideal_profit = ((ideal_exp_np / ideal_enp_np - fee - 1) * p_qtys_np).sum(axis=1) + 1
        real_profit = ((exp_np / enp_np - fee - 1) * p_qtys_np).sum(axis=1) + 1

    sys_log.info("--------- bills ---------")
    sys_log.info("ideal_enp_np : {}".format(ideal_enp_np))
    sys_log.info("ideal_exp_np : {}".format(ideal_exp_np))
    sys_log.info("real_enp : {}".format(enp_np))
    sys_log.info("real_exp : {}".format(exp_np))
    sys_log.info("fee : {}".format(fee))
    # sys_log.info("temp_qty : {}".format(temp_qty))

    return ideal_profit, real_profit

def calc_ideal_profit_v3(self, res_df, open_side, ideal_ep, tp_exec_dict, open_executedPrice_list,
                         tp_executedPrice_list, out_executedPrice_list, fee, trade_log):
    # 1. ideal_ep & tp_exec_dict exist for ideal_pr
    # 2. open_executedPrice_list, close_executedPrice_list exist for real_pr
    #   1. ideal
    #       a. (tp_exec_dict / ideal_ep - fee - 1) * p_qtys (long)
    #           i. dict item -> np.array
    #   2. real
    #   3. open_comparison 생략
    ideal_exp_np = np.hstack(np.array(list(tp_exec_dict.values())))  # list 로 저장된 value, 1d_array 로 conversion
    ideal_enp_np = np.array([ideal_ep] * len(ideal_exp_np))
    # market_close 에서 -2002, -4003 발생한 경우 = error 인데 close 된 경우
    # -> ideal_tp 적용
    if out_executedPrice_list is None:
        out_executedPrice_list = list(ideal_exp_np[-1:])    # list type 사용해야 append 구현됨
    exp_np = np.array(tp_executedPrice_list + out_executedPrice_list)
    enp_np = np.array(open_executedPrice_list * len(exp_np))

    p_qty_ratio_np = np.array(literal_eval(self.config.tp_set.p_qty_ratio))
    if open_side == OrderSide.SELL:
        ideal_profit = ((ideal_enp_np / ideal_exp_np - fee - 1) * p_qty_ratio_np).sum() + 1  # exp_np = 1d
        real_profit = ((enp_np / exp_np - fee - 1) * p_qty_ratio_np).sum() + 1
    else:
        ideal_profit = ((ideal_exp_np / ideal_enp_np - fee - 1) * p_qty_ratio_np).sum() + 1
        real_profit = ((exp_np / enp_np - fee - 1) * p_qty_ratio_np).sum() + 1

    sys_log.info("--------- bills ---------")
    sys_log.info("ideal_enp_np : {}".format(ideal_enp_np))
    sys_log.info("ideal_exp_np : {}".format(ideal_exp_np))
    sys_log.info("real_enp : {}".format(enp_np))
    sys_log.info("real_exp : {}".format(exp_np))
    sys_log.info("fee : {}".format(fee))
    # sys_log.info("temp_qty : {}".format(temp_qty))

    # ------ trade_log ------ #
    for k_ts, tps in tp_exec_dict.items():  # tp_exec_dict 의 사용 의미는 ideal / real_pr 계산인건가
        trade_log[k_ts] = [tps, "exit"]     # Todo, trade_log 사용하는 phase, list 로 변한 tps 주의

    return ideal_profit, real_profit, trade_log

def calc_ideal_profit_v2(self, open_side, ideal_ep, tp_exec_dict, open_executedPrice_list, close_executedPrice_list, res_df, fee, trade_log):
    #   Todo - 추후에 list_comprehension 진행
    #    1. ideal_tp_list 구하기
    #    2. temp_qty_list 와 matching, order_side 기준으로 np.sum(calc_pr_list) 구하기
    #    3. 아직, partial_tp 몇개 체결 후 market_close 에 대한 logic 이 없음 - 현재 일방적 partial_qty 할당됨
    ideal_profit = 1
    real_profit = 1
    r_qty = 1  # base asset ratio
    real_ep = open_executedPrice_list[0]
    if close_executedPrice_list is None:
        close_executedPrice_list = [None] * len(tp_exec_dict)
    # ------------ calc in for loop ------------ #
    #   ts 기준 reverse 정렬, close_executedPrice_list 는 이미 정렬됨 (partial_tp, 가장 먼 tp 부터 pos_res 담김)
    for q_i, ((k_ts, v_tp), real_tp) in enumerate(zip(sorted(tp_exec_dict.items(), key=lambda x: x[0], reverse=True), close_executedPrice_list)):
        # ------ get ideal_tp - dynamic_tp 로 인한 open tp case check ------ #
        if len(v_tp) > 1:
            prev_tp, tp = v_tp
            if prev_tp == tp:
                ideal_tp = tp
            else:
                if open_side == OrderSide.SELL:
                    if tp > res_df['open'].loc[k_ts]:
                        ideal_tp = res_df['open'].loc[k_ts]
                        sys_log.info("market tp executed !")
                    else:
                        ideal_tp = tp
                else:
                    if tp < res_df['open'].loc[k_ts]:
                        ideal_tp = res_df['open'].loc[k_ts]
                        sys_log.info("market tp executed !")
                    else:
                        ideal_tp = tp
        else:  # market close's ideal_tp or out (partial_tp also have market out)
            [ideal_tp] = v_tp

        # ------ check partial_tp case ------ #
        if len(tp_exec_dict) == 1:
            temp_qty = r_qty
        else:
            # if q_i != 0:
            if q_i != len(tp_exec_dict) - 1:
                temp_qty = r_qty / self.config.tp_set.partial_qty_divider
            else:
                temp_qty = r_qty

        r_qty -= temp_qty

        if real_tp is None:
            real_tp = ideal_tp

        if open_side == OrderSide.SELL:
            ideal_profit += (ideal_ep / ideal_tp - fee - 1) * temp_qty
            real_profit += (real_ep / real_tp - fee - 1) * temp_qty
        else:
            ideal_profit += (ideal_tp / ideal_ep - fee - 1) * temp_qty
            real_profit += (real_tp / real_ep - fee - 1) * temp_qty

        sys_log.info("------------ bills ------------")
        sys_log.info("ideal_ep : {}".format(ideal_ep))
        sys_log.info("ideal_tp : {}".format(ideal_tp))
        sys_log.info("real_ep : {}".format(real_ep))
        sys_log.info("real_tp : {}".format(real_tp))
        sys_log.info("fee : {}".format(fee))
        # sys_log.info("temp_qty : {}".format(temp_qty))

        # ------------ save exit data ------------ #
        #       exit_timeindex use "-1" index        #
        # exit_timeindex = str(res_df.index[-1])
        # trade_log[exit_timeindex] = [ideal_tp, "close"]
        trade_log[k_ts] = [ideal_tp, "close"]

    return ideal_profit, real_profit, trade_log
