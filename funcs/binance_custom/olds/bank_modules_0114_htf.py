from funcs.binance.futures_modules import total_income, OrderSide, OrderType, get_market_price_v2, \
    get_order_info, check_exec_by_order_info, get_availableBalance, request_client, sub_client, FuturesMarginType, get_limit_leverage, \
    callback, error
from funcs.binance.futures_concat_candlestick_ftr import concat_candlestick
from funcs.public.broker import calc_rows_and_days
import numpy as np
from datetime import datetime
from easydict import EasyDict
import json
import time
import logging

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


def get_open_side(self, res_df, np_timeidx):
    #        Todo : list comprehension 가능할 것        #
    open_side = None
    for utils_, cfg_ in zip(self.utils_list, self.config_list):
        #       open_score     #
        if res_df['short_open_{}'.format(cfg_.strat_version)][cfg_.trader_set.open_index] == cfg_.ep_set.short_entry_score:
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
        if res_df['long_open_{}'.format(cfg_.strat_version)][cfg_.trader_set.open_index] == \
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


def get_income_info(self, symbol, real_balance, leverage, ideal_profit, start_timestamp, end_timestamp):
    while 1:
        try:
            income = total_income(symbol, start_timestamp, end_timestamp)
            self.accumulated_income += income
            sys_log.info("income : {} USDT".format(income))
        except Exception as e:
            sys_log.error("error in total_income : {}".format(e))
        else:
            break
    tmp_profit = income / real_balance * leverage
    self.accumulated_profit *= (1 + tmp_profit)
    self.calc_accumulated_profit *= 1 + (ideal_profit - 1) * leverage
    sys_log.info('temporary profit : {:.2%} ({:.2%}) %%'.format(tmp_profit, (ideal_profit - 1) * leverage))
    sys_log.info('accumulated profit : {:.2%} ({:.2%}) %%'.format((self.accumulated_profit - 1), (self.calc_accumulated_profit - 1)))
    sys_log.info('accumulated income : {} USDT\n'.format(self.accumulated_income))

    return income, self.accumulated_income, self.accumulated_profit, self.calc_accumulated_profit


def calc_ideal_profit(self, open_side, close_side, tp_exec_dict, res_df, ep, fee, trade_log):
    #   Todo - 추후에 list_comprehension 진행
    #    1. real_tp_list 구하기
    #    2. temp_qty_list 와 matching, order_side 기준으로 np.sum(calc_pr_list) 구하기
    #    3. open_side & close_side 변수 합
    calc_tmp_profit = 1
    r_qty = 1  # base asset ratio
    # ------------ calc in for loop ------------ #
    for q_i, (k_ts, v_tp) in enumerate(sorted(tp_exec_dict.items(), key=lambda x: x[0], reverse=True)):
        # ------ get real_tp ------ #
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
        else:  # market close's real_tp or out (partial_tp also have market out)
            [real_tp] = v_tp

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

        if open_side == OrderSide.SELL:
            calc_tmp_profit += (ep / real_tp - fee - 1) * temp_qty
            # calc_tmp_profit += ep / real_tp - fee
        else:
            calc_tmp_profit += (real_tp / ep - fee - 1) * temp_qty

        sys_log.info("real_tp : {}".format(real_tp))
        sys_log.info("ep : {}".format(ep))
        sys_log.info("fee : {}".format(fee))
        sys_log.info("temp_qty : {}".format(temp_qty))

        # ------------ save exit data ------------ #
        #       exit_timeindex use "-1" index        #
        # exit_timeindex = str(res_df.index[-1])
        # trade_log[exit_timeindex] = [real_tp, "close"]
        trade_log[k_ts] = [real_tp, "close"]

    return calc_tmp_profit, trade_log


def get_new_df(self, calc_rows=True, mode="OPEN"):
    while 1:
        try:
            if calc_rows:
                use_rows, days = calc_rows_and_days(self.config.trader_set.itv_list, self.config.trader_set.row_list,
                                                                            self.config.trader_set.rec_row_list)
            else:
                use_rows, days = self.config.trader_set.row_list[0], 1   # load_new_df3 사용 의도 = close, time logging
            new_df_, _ = concat_candlestick(self.config.trader_set.symbol, '1m',
                                            days=days,
                                            limit=use_rows,
                                            timesleep=None,
                                            show_process=0)
            if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                continue
            if mode == "OPEN":
                sys_log.info("complete new_df_'s time : {}".format(datetime.now()))

        except Exception as e:
            sys_log.error("error in get_new_df : {}".format(e))
            #        1. adequate term for retries
            time.sleep(self.config.trader_set.api_retry_term)
            continue
        else:
            return new_df_, _, 0    # 0 for load_new_df var.


def check_out(self, res_df, market_close_on, log_tp, open_side, out):
    # ------------ open_side 기준 ------------ #
    if open_side == OrderSide.SELL:
        # ------------ out ------------ #
        if self.config.out_set.use_out:
            # ------ short - hl_out ------ #
            if self.config.out_set.hl_out:
                realtime_price = get_market_price_v2(self.sub_client)
                const_str = "realtime_price >= out"
                if eval(const_str):
                    market_close_on = True
                    log_tp = out
                    sys_log.info("{} : {} {}".format(const_str, realtime_price, out))

            # ------ short - close_out ------ #
            else:
                j = self.config.trader_set.complete_index
                if res_df['close'].iloc[j] >= out:
                    market_close_on = True
                    log_tp = res_df['close'].iloc[j]
                    sys_log.info("out : {}".format(out))
    else:
        if self.config.out_set.use_out:
            # ------------ out ------------ #
            # ------ long - hl_out ------ #
            if self.config.out_set.hl_out:
                realtime_price = get_market_price_v2(self.sub_client)
                const_str = "realtime_price <= out"
                if eval(const_str):
                    market_close_on = True
                    log_tp = out
                    sys_log.info("{} : {} {}".format(const_str, realtime_price, out))

            # ------ long - close_out ------ #
            else:
                j = self.config.trader_set.complete_index
                if res_df['close'].iloc[j] <= out:
                    market_close_on = True
                    log_tp = res_df['close'].iloc[j]
                    sys_log.info("out : {}".format(out))

    return market_close_on, log_tp


def check_market_tp(self, res_df, market_close_on, log_tp, cross_on, open_side, tp):
    strat_version = self.config.strat_version
    #       Todo        #
    #        1. 대소 비교 -> np.arr 화 시키면 open_side 1 / 0 으로 구분가능함 (frame 통일시킬 수 있다는 이야기)
    if open_side == OrderSide.SELL:
        # ------------ market_close ------------ #
        # ------ short ------ #
        if self.config.tp_set.tp_type == "MARKET" or self.config.tp_set.tp_type == "BOTH":
            j = self.config.trader_set.complete_index
            # ------ rsi market_close ------ #
            if strat_version in self.config.trader_set.rsi_out_stratver:
                if (res_df['rsi_%s' % self.config.loc_set.point.exp_itv].iloc[
                        j - 1] >= 50 - self.config.loc_set.point.osc_band) & \
                        (res_df['rsi_%s' % self.config.loc_set.point.exp_itv].iloc[
                             j] < 50 - self.config.loc_set.point.osc_band):
                    market_close_on = True

            # ------ early_out ------ #
            if strat_version in ['v5_2']:
                if res_df['close'].iloc[j] < res_df['bb_lower_5m'].iloc[j] < \
                        res_df['close'].iloc[j - 1]:
                    cross_on = 1
                if cross_on == 1 and res_df['close'].iloc[j] > res_df['bb_upper_5m'].iloc[j] > \
                        res_df['close'].iloc[j - 1]:
                    market_close_on = True

            # ------ time_tp ------ # - income() waiting issue 로 complete_index 사용함
            if self.config.tp_set.time_tp:
                if res_df['short_close_{}'.format(self.config.strat_version)][j] == self.config.ep_set.short_entry_score:
                    market_close_on = True

            if market_close_on:
                log_tp = res_df['close'].iloc[j]
                sys_log.info("tp : {}".format(tp))
    else:
        # ------ long ------ #
        if self.config.tp_set.tp_type == "MARKET" or self.config.tp_set.tp_type == "BOTH":
            j = self.config.trader_set.complete_index
            # ------ rsi market_close ------ #
            if strat_version in self.config.trader_set.rsi_out_stratver:
                if (res_df['rsi_%s' % self.config.loc_set.point.exp_itv].iloc[
                        j - 1] <= 50 + self.config.loc_set.point.osc_band) & \
                        (res_df['rsi_%s' % self.config.loc_set.point.exp_itv].iloc[
                             j] > 50 + self.config.loc_set.point.osc_band):
                    market_close_on = True

            # ------ early_out ------ #
            if strat_version in ['v5_2']:
                if res_df['close'].iloc[j] > res_df['bb_upper_5m'].iloc[j] > \
                        res_df['close'].iloc[j - 1]:
                    cross_on = 1
                if cross_on == 1 and res_df['close'].iloc[j] < res_df['bb_lower_5m'].iloc[j] < \
                        res_df['close'].iloc[j - 1]:
                    market_close_on = True

            # ------ time_tp ------ # - income() waiting issue 로 complete_index 사용함
            if self.config.tp_set.time_tp:
                if res_df['long_close_{}'.format(self.config.strat_version)][j] == -self.config.ep_set.short_entry_score:
                    market_close_on = True

            if market_close_on:
                log_tp = res_df['close'].iloc[j]
                sys_log.info("tp : {}".format(tp))

    return market_close_on, log_tp, cross_on


def check_limit_tp_exec(self, post_order_res_list, quantity_precision):
    if self.config.tp_set.tp_type == OrderType.LIMIT or self.config.tp_set.tp_type == "BOTH":
        order_exec_check_list = [check_exec_by_order_info(self.config.trader_set.symbol, post_order_res, quantity_precision)
                                 for post_order_res in post_order_res_list if post_order_res is not None]

        if np.sum(order_exec_check_list) == len(order_exec_check_list):
            sys_log.info('all limit tp order executed')
            return 1
    return 0


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
        tp = tp_series.iloc[self.config.trader_set.complete_index]
    if not self.config.out_set.static_out:
        out = out_series.iloc[self.config.trader_set.complete_index]

    return tp, out, tp_series, out_series


def check_breakout_qty(self, first_exec_qty_check, check_time, post_order_res, open_quantity):
    #        1. breakout_gty_ratio
    #           a. exec_open_qty 와 open_quantity 의 ratio 비교
    #           b. realtime_term 에 delay 주지 않기 위해, term 설정
    #        2. 이걸 설정 안해주면 체결되도 close_order 진행 안됨
    if first_exec_qty_check or time.time() - check_time >= self.config.trader_set.qty_check_term:
        while 1:
            try:
                order_info_res = get_order_info(self.config.trader_set.symbol, post_order_res.orderId)
                open_executedQty = float(order_info_res.executedQty)
                breakout = abs(open_executedQty / open_quantity) >= self.config.trader_set.breakout_qty_ratio
            except Exception as e:
                sys_log.error('error in check_breakout_qty : {}'.format(e))
                time.sleep(self.config.trader_set.api_retry_term)
                continue
            else:
                return 0, time.time(), breakout  # first_exec_qty_check, check_time, breakout
    return 0, check_time, 0  # breakout_qty survey 실행한 경우만 check_time 갱신


def check_ei_k(self, res_df_init, open_side):
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
        tp_j = self.config.trader_set.complete_index
        if open_side == OrderSide.SELL:
            if realtime_price <= res_df_init['h_short_rtc_1_{}'.format(strat_version)].iloc[tp_j] - \
                    res_df_init['h_short_rtc_gap_{}'.format(strat_version)].iloc[
                        tp_j] * self.config.loc_set.zone.ei_k:
                sys_log.info("cancel open_order by ei_k\n")
                ep_out = 1
        else:
            if realtime_price >= res_df_init['h_long_rtc_1_{}'.format(strat_version)].iloc[tp_j] + \
                    res_df_init['h_long_rtc_gap_{}'.format(strat_version)].iloc[
                        tp_j] * self.config.loc_set.zone.ei_k:
                sys_log.info("cancel open_order by ei_k\n")
                ep_out = 1
        time.sleep(self.config.trader_set.realtime_term)  # <-- for realtime price function

    return ep_out


def check_ei_k_onclose(self, res_df_init, res_df, open_side, e_j, tp_j):
    ep_out = 0
    strat_version = self.config.strat_version
    if open_side == OrderSide.SELL:
        if res_df['low'].iloc[e_j] <= res_df_init['h_short_rtc_1_{}'.format(strat_version)].iloc[tp_j] - \
                res_df_init['h_short_rtc_gap_{}'.format(strat_version)].iloc[tp_j] * self.config.loc_set.zone.ei_k:
            sys_log.info("cancel open_order by ei_k\n")
            ep_out = 1

    else:
        if res_df['high'].iloc[e_j] >= res_df_init['h_long_rtc_1_{}'.format(strat_version)].iloc[tp_j] + \
                res_df_init['h_long_rtc_gap_{}'.format(strat_version)].iloc[tp_j] * self.config.loc_set.zone.ei_k:
            sys_log.info("cancel open_order by ei_k\n")
            ep_out = 1

    return ep_out


def get_balance(self, first_iter, cfg_path_list):
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
    while 1:
        try:
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
                return self.available_balance, self.over_balance, 1

        except Exception as e:
            sys_log.error('error in get_availableBalance : {}\n'.format(e))
            time.sleep(self.config.trader_set.api_retry_term)
            continue
        else:
            return self.available_balance, self.over_balance, 0


def get_eptpout(self, e_j, open_side, res_df_init, res_df):
    sys_log.info('open_side : {}'.format(open_side))
    strat_version = self.config.strat_version
    if strat_version in ['v5_2']:
        out_j = e_j
        # ep_j = e_j
    else:
        out_j = self.config.trader_set.complete_index
    ep_j = self.config.trader_set.complete_index
    tp_j = self.config.trader_set.complete_index

    #        1. 추후 other strat. 적용시 res_df_init recheck
    #       Todo        #
    #           b. --> res_df 사용시, dynamic (pre, whole) 을 허용하겠다는 의미
    #               i. ep_loc_point2 를 사용하지 않는 이상 res_df = res_df_init
    if open_side == OrderSide.BUY:
        ep = res_df_init['long_ep_{}'.format(strat_version)].iloc[ep_j]

        if self.config.pos_set.long_inversion:
            open_side = OrderSide.SELL  # side change (inversion)
            out = res_df['long_tp_{}'.format(strat_version)].iloc[out_j]
            tp = res_df_init['long_out_{}'.format(strat_version)].iloc[tp_j]
        else:
            out = res_df['long_out_{}'.format(strat_version)].iloc[out_j]
            tp = res_df_init['long_tp_{}'.format(strat_version)].iloc[tp_j]
    else:
        ep = res_df_init['short_ep_{}'.format(strat_version)].iloc[ep_j]

        if self.config.pos_set.short_inversion:
            open_side = OrderSide.BUY
            out = res_df['short_tp_{}'.format(strat_version)].iloc[out_j]
            tp = res_df_init['short_out_{}'.format(strat_version)].iloc[tp_j]
        else:
            out = res_df['short_out_{}'.format(strat_version)].iloc[out_j]
            tp = res_df_init['short_tp_{}'.format(strat_version)].iloc[tp_j]

    return ep, tp, out, open_side


