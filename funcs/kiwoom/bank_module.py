from funcs.kiwoom.kiwoom_module import KiwoomModule
from funcs.public.broker import calc_rows_and_days
from funcs.public.constant import *

import numpy as np
import pandas as pd
from datetime import datetime
import time
from ast import literal_eval
import pickle
import logging.config


class BankModule(KiwoomModule):

    def __init__(self, config):

        secret_key = self.load_key(r"D:\Projects\System_Trading\JnQ_32bit\Bank\api_keys\kiwoom_5189140110.pkl")
        super().__init__(secret_key)

        # 0. config, sys_log 는 BankModule 에서만 사용됨.
        self.config = config
        self.sys_log = logging.getLogger()

    @staticmethod
    def load_key(key_abspath):
        with open(key_abspath, 'rb') as f:
            return pickle.load(f)

    def get_new_df_v2(self, code, api_term=0.3, calc_rows=True, mode="OPEN"):

        """
        v1 -> v2
        Bank() 환경에 맞추어 return value 1개로 설정.
        1. remove use_rows : 사용하지 않아서.
        2. last row validation
            a. 검수를 위해 지금은 continue 풀어놓음.
        """

        while 1:
            try:
                row_list = literal_eval(self.config.trader_set.row_list)
                if calc_rows:
                    itv_list = literal_eval(self.config.trader_set.itv_list)
                    rec_row_list = literal_eval(self.config.trader_set.rec_row_list)
                    _, days = calc_rows_and_days(itv_list, row_list, rec_row_list)
                else:
                    _, days = row_list[0], 1  # load_new_df3 사용 의도 = close, time logging
                new_df_ = self.concat_candlestick(days, api_term, 종목코드=code, 틱범위="1", 수정주가구분="0")

                # new_df_ validity check (Stock 의 경우 59.999 초를 더해주어야함, timeindex 의 second = 0)
                #    1. datetime timestamp() 의 기준은 second 임.
                if datetime.timestamp(new_df_.index[-1]) + 59.999 < datetime.now().timestamp():
                    self.sys_log.warning("dataframe last row validation warning, last index : {}".format(str(new_df_.index[-1])))
                    # time.sleep(self.config.trader_set.api_term)
                    # continue
                if mode == "OPEN":
                    self.sys_log.info("complete new_df_'s time : {}".format(datetime.now()))

            except Exception as e:
                self.sys_log.error("error in get_new_df : {}".format(e))
                #        1. adequate term for retries
                time.sleep(self.config.trader_set.api_term)
                continue
            else:
                return new_df_  # 0 for load_new_df var.

    def get_streamer(self, code):

        """
        back_data_path 에 code 로 입력할 것
        """

        row_list = literal_eval(self.config.trader_set.row_list)
        itv_list = literal_eval(self.config.trader_set.itv_list)
        rec_row_list = literal_eval(self.config.trader_set.rec_row_list)
        use_rows, days = calc_rows_and_days(itv_list, row_list, rec_row_list)

        # back_df = pd.read_pickle(self.config.trader_set.back_data_path.replace("code", code))
        back_df = pd.read_pickle(self.config.trader_set.back_data_path.replace("code", code)).astype(float)  # int 로 저장해서, 임시적으로 변경함.

        if self.config.trader_set.start_datetime != "None":
            target_datetime = pd.to_datetime(self.config.trader_set.start_datetime)
            if target_datetime in back_df.index:
                start_idx = np.argwhere(back_df.index == target_datetime).item()
            else:
                start_idx = use_rows
        else:
            start_idx = use_rows

        # 시작하는 idx 는 필요로하는 rows 보다 커야함.
        assert start_idx >= use_rows, "more dataframe rows required"

        for i in range(start_idx + 1, len(back_df)):  # +1 for i inclusion
            yield back_df.iloc[i - use_rows:i]

    def get_new_df_onstream_v2(self, streamer):
        """
        v1 -> v2
            1. return value 1개로 설정.
        """

        try:
            res_df = next(streamer)  # Todo, 무결성 검증 미진행
        except Exception as e:
            self.sys_log.error("error in get_new_df_onstream : {}".format(e))
            quit()  # error 처리할 필요없이, backtrader 프로그램 종료
            # return None, None   # output len 유지
        else:
            return res_df

    def get_open_side_v2(self, res_df, papers, np_timeidx, open_num=1):

        public, utils_list, config_list = papers
        utils, config = None, config_list[0]    # 첫 config 로 초기화.
        open_side = None

        for utils_, cfg_ in zip(utils_list, config_list):

            # ------ 1. point ------ #
            #  여기 왜 latest_index 사용했는지 모르겠음 -> bar_close point 를 고려해 mr_res 와의 index 분리
            #  vecto. 하는 이상, point x ep_loc 의 index 는 sync. 되어야함 => complete_index 사용 (solved)
            #  각 phase 분리한 이유 = sys_logging (solved)
            if res_df['short_open{}_{}'.format(open_num, cfg_.selection_id)].to_numpy()[cfg_.trader_set.complete_index]:
                self.sys_log.warning("[ short ] true_point selection_id : {}".format(cfg_.selection_id))

                # ------ 2. ban ------ #
                if not cfg_.pos_set.short_ban:

                    # ------ 3. mr_res ------ #
                    if open_num == 1:
                        mr_res, zone_arr = public.ep_loc_p1_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL)
                    else:
                        mr_res, zone_arr = public.ep_loc_p2_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL)
                    self.sys_log.warning("mr_res[cfg_.trader_set.complete_index] : {}".format(mr_res[cfg_.trader_set.complete_index]))

                    # ------ assign ------ #
                    if mr_res[cfg_.trader_set.complete_index]:
                        utils = utils_
                        config = cfg_
                        open_side = OrderSide.SELL
                        break
                else:
                    self.sys_log.warning("cfg_.pos_set.short_ban : {}".format(cfg_.pos_set.short_ban))

            # ------ 1. point ------ #
            #       'if' for nested ep_loc.point        #
            if res_df['long_open{}_{}'.format(open_num, cfg_.selection_id)].to_numpy()[cfg_.trader_set.complete_index]:
                self.sys_log.warning("[ long ] true_point selection_id : {}".format(cfg_.selection_id))

                # ------ 2. ban ------ #
                if not cfg_.pos_set.long_ban:  # additional const. on trader

                    # ------ 3. mr_res ------ #
                    if open_num == 1:
                        mr_res, zone_arr = public.ep_loc_p1_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY)
                    else:
                        mr_res, zone_arr = public.ep_loc_p2_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY)
                    self.sys_log.warning("mr_res[cfg_.trader_set.complete_index] : {}".format(mr_res[cfg_.trader_set.complete_index]))

                    # ------ assign ------ #
                    if mr_res[cfg_.trader_set.complete_index]:
                        utils = utils_
                        config = cfg_
                        open_side = OrderSide.BUY
                        break
                else:
                    self.sys_log.warning("cfg_.pos_set.long_ban : {}".format(cfg_.pos_set.long_ban))

        return open_side, utils, config

    def get_tpepout(self, open_side, res_df_open, res_df, mode="ep1"):
        self.sys_log.info('open_side : {}'.format(open_side))
        selection_id = self.config.selection_id

        tp_j = self.config.trader_set.complete_index
        ep_j = self.config.trader_set.complete_index
        #  complete_index + 1 이 아니라, 해당 index 롤 res_df_open 이 아닌 -> res_df 에 입력하면 되는거아닌가
        #  -> use_point2 여부에 따라 다르게 작성할 필요가 없어짐 (solved)
        out_j = self.config.trader_set.complete_index

        #        b. --> res_df 사용시, dynamic (pre, whole) 을 허용하겠다는 의미
        #           i. ep_loc_point2 미사용시, res_df = res_df_open
        if open_side == OrderSide.BUY:
            ep = res_df_open['long_ep1_{}'.format(selection_id)].to_numpy()[ep_j]

            if self.config.pos_set.long_inversion:
                open_side = OrderSide.SELL  # side change (inversion)
                out = res_df_open['long_tp_{}'.format(selection_id)].to_numpy()[out_j]
                tp = res_df_open['long_out_{}'.format(selection_id)].to_numpy()[tp_j]
            else:
                out = res_df_open['long_out_{}'.format(selection_id)].to_numpy()[out_j]
                tp = res_df_open['long_tp_{}'.format(selection_id)].to_numpy()[tp_j]
        else:
            ep = res_df_open['short_ep1_{}'.format(selection_id)].to_numpy()[ep_j]

            if self.config.pos_set.short_inversion:
                open_side = OrderSide.BUY
                out = res_df_open['short_tp_{}'.format(selection_id)].to_numpy()[out_j]
                tp = res_df_open['short_out_{}'.format(selection_id)].to_numpy()[tp_j]
            else:
                out = res_df_open['short_out_{}'.format(selection_id)].to_numpy()[out_j]
                tp = res_df_open['short_tp_{}'.format(selection_id)].to_numpy()[tp_j]

        return tp, ep, out, open_side

    def check_open_exec_qty_v2(self, order_no):
        """
        1. breakout_gty_ratio
            a. exec_open_qty 와 open_quantity 의 ratio 비교
            b. realtime_term 에 delay 주지 않기 위해, term 설정
        2. 이걸 설정 안해주면 체결되도 close_order 진행 안됨
        """

        order_info = self.get_order_info(계좌번호=self.account_number, 전체종목구분=0, 매매구분="0", 종목코드="", 체결구분="0")
        order_info_valid = order_info.loc[order_no]
        open_exec = order_info_valid['체결량'] / order_info_valid['주문수량'] >= self.config.trader_set.open_exec_qty_ratio \
                    or order_info_valid['주문상태'] == "체결"
        return open_exec

    def check_ei_k_v2(self, res_df_open, res_df, realtime_price, open_side):

        """
        = expire_tp
            1. check_ei_k_onbarclose_v2 에서는 expire_tick 기준으로 res_df_open, res_df 둘다 사용함.
                a. 형태 유지를 위해 이곳에서도 res_df input 으로 유지함.
            2. remove realtime term -> 외부에서 API limit 을 위한 term 관리를 진행함.
        """

        ep_out = 0
        selection_id = self.config.selection_id

        if self.config.tr_set.expire_tick != "None":
            if time.time() - datetime.timestamp(res_df_open.index[self.config.trader_set.latest_index]) \
                    >= self.config.tr_set.expire_tick * 60:
                ep_out = 1

        if self.config.tr_set.expire_k1 != "None":  # Todo -> 이거 None 이면 "무한 대기" 발생 가능함
            # realtime_price = self.get_market_price_v2()
            tp_j = self.config.trader_set.complete_index
            #       Todo        #
            #        2. 추후, dynamic_tp 사용시 res_df 갱신해야할 것
            #           a. 그에 따른 res_df 종속 변수 check
            #        3. ei_k - ep_out 변수 달아주고, close bar waiting 추가할지 고민중
            #        4. warning - 체결량 존재하는데, ep_out 가능함 - market 고민중
            #        5. funcs_trader_modules 에서 order_side check 하는 function 모두 inversion 고려해야할 것
            #           a. "단, order_side change 후로만 해당됨
            if open_side == OrderSide.SELL:
                short_tp_ = res_df_open['short_tp_{}'.format(selection_id)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
                short_tp_gap_ = res_df_open['short_tp_gap_{}'.format(selection_id)].to_numpy()
                if realtime_price <= short_tp_[tp_j] + short_tp_gap_[tp_j] * self.config.tr_set.expire_k1:
                    ep_out = 1
            else:
                long_tp_ = res_df_open['long_tp_{}'.format(selection_id)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
                long_tp_gap_ = res_df_open['long_tp_gap_{}'.format(selection_id)].to_numpy()
                if realtime_price >= long_tp_[tp_j] - long_tp_gap_[tp_j] * self.config.tr_set.expire_k1:
                    ep_out = 1

        if ep_out:
            self.sys_log.warning("cancel open_order by ei_k\n")

        return ep_out

    def check_ei_k_onbarclose_v2(self, res_df_open, res_df, e_j, tp_j, open_side):  # for point2

        """
        = expire_tp
        """

        ep_out = 0
        selection_id = self.config.selection_id

        if self.config.tr_set.expire_tick != "None":
            if datetime.timestamp(res_df.index[-1]) - datetime.timestamp(res_df_open.index[self.config.trader_set.latest_index]) \
                    >= self.config.tr_set.expire_tick * 60:
                ep_out = 1

        if self.config.tr_set.expire_k1 != "None":  # Todo - onbarclose 에서는, res_df_open 으로 open_index 의 tp 정보를 사용
            if open_side == OrderSide.SELL:
                low = res_df['low'].to_numpy()
                short_tp_ = res_df_open['short_tp_{}'.format(selection_id)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
                short_tp_gap_ = res_df_open['short_tp_gap_{}'.format(selection_id)].to_numpy()
                if low[e_j] <= short_tp_[tp_j] + short_tp_gap_[tp_j] * self.config.tr_set.expire_k1:
                    ep_out = 1
            else:
                high = res_df['high'].to_numpy()
                long_tp_ = res_df_open['long_tp_{}'.format(selection_id)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
                long_tp_gap_ = res_df_open['long_tp_gap_{}'.format(selection_id)].to_numpy()
                if high[e_j] >= long_tp_[tp_j] - long_tp_gap_[tp_j] * self.config.tr_set.expire_k1:
                    ep_out = 1

        if ep_out:
            self.sys_log.warning("cancel open_order by ei_k\n")

        return ep_out

    def get_dynamic_tpout(self, res_df, open_side, tp, out):
        # ------ dynamic & inversion tp_out ------ #
        selection_id = self.config.selection_id
        if open_side == OrderSide.BUY:
            if self.config.pos_set.short_inversion:
                out_series = res_df['short_tp_{}'.format(selection_id)]
                tp_series = res_df['short_out_{}'.format(selection_id)]
            else:
                out_series = res_df['long_out_{}'.format(selection_id)]
                tp_series = res_df['long_tp_{}'.format(selection_id)]
        else:
            if self.config.pos_set.long_inversion:
                out_series = res_df['long_tp_{}'.format(selection_id)]
                tp_series = res_df['long_out_{}'.format(selection_id)]
            else:
                out_series = res_df['short_out_{}'.format(selection_id)]
                tp_series = res_df['short_tp_{}'.format(selection_id)]

        if not self.config.tp_set.static_tp:
            tp = tp_series.to_numpy()[self.config.trader_set.complete_index]
        if not self.config.out_set.static_out:
            out = out_series.to_numpy()[self.config.trader_set.complete_index]

        return tp, out, tp_series, out_series

    def get_partial_tp_qty(self, ep, tp, open_exec_qty, quantity_precision, close_side):

        """
        changed to calc_with_precision -> calc_with_hoga_unit for price, in Stock
        """
        partial_ranges, partial_qty_ratio = literal_eval(self.config.tp_set.partial_ranges), literal_eval(self.config.tp_set.partial_qty_ratio)
        partial_length = len(partial_ranges)
        assert partial_length == len(partial_qty_ratio)
        en_ps, tps, open_qtys = [np.tile(arr_, (partial_length,)) for arr_ in [ep, tp, open_exec_qty]]

        if close_side == OrderSide.BUY:  # short
            partial_tps = en_ps - (en_ps - tps) * partial_ranges
        else:  # long
            partial_tps = en_ps + (tps - en_ps) * partial_ranges
        partial_qtys = open_qtys * partial_qty_ratio

        partial_tps = list(map(lambda x: self.calc_with_hoga_unit(x), partial_tps))
        partial_qtys = list(map(lambda x: self.calc_with_precision(x, quantity_precision), partial_qtys))
        self.sys_log.info('partial_tps : {}'.format(partial_tps))
        self.sys_log.info('partial_qtys : {}'.format(partial_qtys))

        return partial_tps, partial_qtys

    def check_limit_tp_exec_v2(self, order_no_list, quantity_precision=0):

        all_executed = 0
        order_info = self.get_order_info(계좌번호=self.account_number, 전체종목구분=0, 매매구분="0", 종목코드="", 체결구분="0")

        #     1. order_no_list 의 "" (공백) 에 대한 무결성 검증해야함. (loc 접근할때 안되니까)
        #         a. 입력되는 order_no_list 에 공백만 존재할 수는 없다고 어느정도 가정한 상황임.
        order_no_arr = np.array(order_no_list)
        order_no_arr = order_no_arr[np.where(order_no_arr != "")[0]]

        order_info_valid = order_info.loc[order_no_arr]
        order_exec_check = order_info_valid["주문수량"] - order_info_valid["체결량"] < 1 / (10 ** quantity_precision)

        # 손매매로 인한 order_info.status == EXPIRED 는 고려 대상이 아님. (손매매 이왕이면 하지 말라는 이야기)
        if np.sum(order_exec_check) == len(order_exec_check):
            self.sys_log.info('all limit tp order executed')
            all_executed = 1

        exec_price_list = [exec_price for exec_price, executed in zip(order_info_valid["체결가"], order_exec_check) if executed]
        return all_executed, exec_price_list

    def check_hl_out_onbarclose_v2(self, res_df, market_close_on, out, liqd_p, open_side):

        """
        v1 -> v2
            1. add liquidation platform.
            2. add log_out
        """

        log_out = None
        c_i = self.config.trader_set.complete_index

        high = res_df['high'].to_numpy()[c_i]
        low = res_df['low'].to_numpy()[c_i]
        close = res_df['close'].to_numpy()[c_i]

        # ------ 1. liquidation default check ------ #
        if open_side == OrderSide.SELL:
            const_str = "high >= liqd_p"
            if eval(const_str):
                market_close_on = True
                log_out = liqd_p
                self.sys_log.info("{} : {} {}".format(const_str, high, liqd_p))
        else:
            const_str = "low <= liqd_p"
            if eval(const_str):
                market_close_on = True
                log_out = liqd_p
                self.sys_log.info("{} : {} {}".format(const_str, low, liqd_p))

        # ------ 2. hl_out ------ #
        if self.config.out_set.hl_out:
            if open_side == OrderSide.SELL:
                const_str = "high >= out"
                if eval(const_str):
                    market_close_on = True
                    log_out = out
                    self.sys_log.info("{} : {} {}".format(const_str, high, out))
            else:
                const_str = "low <= out"
                if eval(const_str):
                    market_close_on = True
                    log_out = out
                    self.sys_log.info("{} : {} {}".format(const_str, low, out))

        # ------ 3. close_out ------ #
        else:
            if open_side == OrderSide.SELL:
                if close >= out:
                    market_close_on = True
                    log_out = close
                    self.sys_log.info("{} : {} {}".format("close >= out", log_out, out))
            else:
                if close <= out:
                    market_close_on = True
                    log_out = close
                    self.sys_log.info("{} : {} {}".format("close <= out", log_out, out))

        return market_close_on, log_out

    def check_hl_out_v3(self, res_df, market_close_on, out, liqd_p, realtime_price, open_side):

        """
        v2 -> v3
            1. get realtime_price from outer_scope
            2. inversion 고려 아직임. (inversion 사용하게 되면 고려할 것.) (Todo)
            3. add log_out
        """

        log_out = None

        # ------ 1. liquidation default check ------ #
        if open_side == OrderSide.SELL:
            const_str = "realtime_price >= liqd_p"
            if eval(const_str):
                market_close_on = True
                log_out = liqd_p
                self.sys_log.info("{} : {} {}".format(const_str, realtime_price, liqd_p))
        else:
            const_str = "realtime_price <= liqd_p"
            if eval(const_str):
                market_close_on = True
                log_out = liqd_p
                self.sys_log.info("{} : {} {}".format(const_str, realtime_price, liqd_p))

        # ------ 2. hl_out ------ #
        if self.config.out_set.hl_out:
            if open_side == OrderSide.SELL:
                const_str = "realtime_price >= out"
                if eval(const_str):
                    market_close_on = True
                    log_out = out
                    self.sys_log.info("{} : {} {}".format(const_str, realtime_price, out))
            else:
                const_str = "realtime_price <= out"
                if eval(const_str):
                    market_close_on = True
                    log_out = out
                    self.sys_log.info("{} : {} {}".format(const_str, realtime_price, out))

        # ------ 3. close_out ------ #
        else:
            close = res_df['close'].to_numpy()

            if open_side == OrderSide.SELL:
                j = self.config.trader_set.complete_index
                if close[j] >= out:
                    market_close_on = True
                    log_out = close[j]
                    self.sys_log.info("{} : {} {}".format("close >= out", log_out, out))
            else:
                j = self.config.trader_set.complete_index
                if close[j] <= out:
                    market_close_on = True
                    log_out = close[j]
                    self.sys_log.info("{} : {} {}".format("close <= out", log_out, out))

        return market_close_on, log_out

    def calc_ideal_profit_v4(self, res_df, open_side, ideal_ep, tp_exec_dict, open_exec_price_list, tp_exec_price_list, out_exec_price_list, fee, trade_log_dict):

        """
        v3 -> v4
            1. 수정된 수수료 계산법 도입
                a. ((ideal_enp_np / ideal_exp_np * (1 - fee) - 1) * partial_qty_ratio_np).sum() + 1
        """
        
        ideal_exp_np = np.hstack(np.array(list(tp_exec_dict.values())))  # list 로 저장된 value, 1d_array 로 conversion
        ideal_enp_np = np.array([ideal_ep] * len(ideal_exp_np))
        # market_close 에서 -2002, -4003 발생한 경우 = error 인데 close 된 경우
        # -> ideal_tp 적용
        if out_exec_price_list is None:
            out_exec_price_list = list(ideal_exp_np[-1:])  # list type 사용해야 append 구현됨
        exp_np = np.array(tp_exec_price_list + out_exec_price_list)
        enp_np = np.array(open_exec_price_list * len(exp_np))

        partial_qty_ratio_np = np.array(literal_eval(self.config.tp_set.partial_qty_ratio))
        if open_side == OrderSide.SELL:
            ideal_profit = ((ideal_enp_np / ideal_exp_np * (1 - fee) - 1) * partial_qty_ratio_np).sum() + 1
            real_profit = ((enp_np / exp_np * (1 - fee) - 1) * partial_qty_ratio_np).sum() + 1
        else:
            ideal_profit = ((ideal_exp_np / ideal_enp_np * (1 - fee) - 1) * partial_qty_ratio_np).sum() + 1
            real_profit = ((exp_np / enp_np * (1 - fee) - 1) * partial_qty_ratio_np).sum() + 1

        self.sys_log.info("--------- bills ---------")
        self.sys_log.info("ideal_enp_np : {}".format(ideal_enp_np))
        self.sys_log.info("ideal_exp_np : {}".format(ideal_exp_np))
        self.sys_log.info("real_enp : {}".format(enp_np))
        self.sys_log.info("real_exp : {}".format(exp_np))
        self.sys_log.info("fee : {}".format(fee))
        # sys_log.info("temp_qty : {}".format(temp_qty))

        # ------ trade_log_dict ------ #
        for k_ts, tps in tp_exec_dict.items():
            trade_log_dict["exit"] = [k_ts, open_side, tps]

        return ideal_profit, real_profit, trade_log_dict

    def check_signal_out_v3(self, res_df, market_close_on, open_side):

        """
        v1 -> v2
            1. phase 별로 정리 (open_side phase 가 하위로 가도록.)
            2. Todo, inversion 에 대한 고려 진행된건가 - 안된것으로 보임
            3. add log_out
            4. remove cross_on
        """

        log_out = None
        c_i = self.config.trader_set.complete_index

        close = res_df['close'].to_numpy()[c_i]

        # ------ 1. rsi_exit ------ #
        if self.config.out_set.rsi_exit:
            rsi_ = res_df['rsi_%s' % self.config.loc_set.point.exp_itv].to_numpy()
            osc_band = self.config.loc_set.point.osc_band

            if open_side == OrderSide.SELL:
                if (rsi_[c_i - 1] >= 50 - osc_band) & (rsi_[c_i] < 50 - osc_band):
                    market_close_on = True
            else:
                if (rsi_[c_i - 1] <= 50 + osc_band) & (rsi_[c_i] > 50 + osc_band):
                    market_close_on = True

        # ------ 2. cci_exit ------ #
        #           a. deprecated.
        # if self.config.out_set.cci_exit:
        #     wave_itv1 = self.config.tr_set.wave_itv1
        #     wave_period1 = self.config.tr_set.wave_period1
        #
        #     if open_side == OrderSide.SELL:
        #         wave_co_ = res_df['wave_co_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[c_i]
        #         if wave_co_:
        #             market_close_on = True
        #     else:
        #         wave_cu_ = res_df['wave_cu_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[c_i]
        #         if wave_cu_:
        #             market_close_on = True

        if market_close_on:
            log_out = close
            self.sys_log.info("signal out : {}".format(log_out))

        return market_close_on, log_out

    def limit_order_v2(self, order_info, order_data=None, **kwargs):

        if order_data is None:
            over_balance = None
        else:
            over_balance, leverage = order_data

        order_side = kwargs['order_side']
        if order_side == "BUY":
            order_side_number = 1
            order_state = "+매수"
        else:
            order_side_number = 2
            order_state = "-매도"

        while 1:
            try:
                self.send_order(kwargs['request_name'], self.screen_number, self.account_number,
                                order_side_number, kwargs['code'], kwargs['qty'], kwargs['price'], "00", "")

            except Exception as e:
                self.sys_log.error('error in limit_order : {}'.format(e))

                # 1. error 처리 (Todo)
                #       a. unknown error : return (retry 가 의미 없을테니까)
                # return "", None, "unknown"  # 출력 형태 유지.
                #       b. known error : retry
                #               i. quantity less than zero
                #               ii. Precision is over the maximum defined for this asset
                #               iii. price precision
                #               iv. Margin is insufficient
                time.sleep(self.config.trader_set.order_term)
                continue

            else:
                if self.order_no == "":
                    self.sys_log.error("limit_order : self.order_no == "", maybe quantity for this order already set.")
                    order_no_list = order_info.index[(order_info.종목코드 == kwargs['code']) & (order_info.주문상태 == "접수") & (order_info.주문상태 == order_state)].tolist()
                    #   a. 주문상태 == "접수" => 미체결 상태의 주문만 cancel 진행함.
                    if len(order_no_list) != 0:
                        try:
                            self.cancel_order_list(kwargs['code'], kwargs['order_side'], order_no_list, order_info)
                        except Exception as e:
                            self.sys_log.error('error in cancel_order_list (limit_order phase) : {}'.format(e))
                        time.sleep(self.config.trader_set.order_term)
                        continue
                    #   b. 주문 단가가 상한선을 초과하는 경우.주문 가능해질때까지 keep.
                    else:
                        return self.order_no, over_balance, 1  # , limit 은 order_no = "" 허용하지 않음.
                else:
                    self.sys_log.info('limit_order enlisted. : {}'.format(datetime.now()))
                    return self.order_no, over_balance, 0  # , error_code

    def limit_order(self, order_data=None, **kwargs):

        if order_data is None:
            over_balance = None
        else:
            over_balance, leverage = order_data

        order_side = kwargs['order_side']
        if order_side == "BUY":
            order_side_number = 1
        else:
            order_side_number = 2

        while 1:
            try:
                self.send_order(kwargs['request_name'], self.screen_number, self.account_number,
                                order_side_number, kwargs['code'], kwargs['qty'], kwargs['price'], "00", "")

            except Exception as e:
                self.sys_log.error('error in limit_order : {}'.format(e))

                # 1. error 처리 (Todo)
                #       a. unknown error : return (retry 가 의미 없을테니까)
                # return "", None, "unknown"  # 출력 형태 유지.
                #       b. known error : retry
                #               i. quantity less than zero
                #               ii. Precision is over the maximum defined for this asset
                #               iii. price precision
                #               iv. Margin is insufficient
                time.sleep(self.config.trader_set.order_term)
                continue

            else:
                if self.order_no == "":
                    self.sys_log.error("limit_order : self.order_no == "", maybe quantity for this order already set.")
                    # time.sleep(self.config.trader_set.order_term)
                    # continue
                    return self.order_no, over_balance, 1  # , limit 은 order_no = "" 허용하지 않음.
                else:
                    self.sys_log.info('limit_order enlisted. : {}'.format(datetime.now()))
                    return self.order_no, over_balance, 0  # , error_code

    def partial_limit_order_v4(self, code, order_no_list, order_info, partial_tps, partial_qtys, close_side, open_exec_qty, quantity_precision):

        """
        v3 -> v4
            1. add. error_code
            2. use limit_order_v2
            3. order_no append -> indexing for keeping invalid order.
        """
        partial_tp_idx = 0
        partial_length = len(partial_tps)
        error_code = 0
        while 1:
            if partial_tp_idx >= partial_length:
                break

            #       x. valid partial_order rejection.
            if order_no_list[partial_tp_idx] != "":
                partial_tp_idx += 1
                continue

            partial_tp = partial_tps[partial_tp_idx]
            #       y. 맨 마지막 잔여 partial_qty => [:-1] indexing 을 사용해 계산함.
            if partial_tp_idx == partial_length - 1:
                partial_qty = self.calc_with_precision(open_exec_qty - np.sum(partial_qtys[:-1]), quantity_precision)
            else:
                partial_qty = partial_qtys[partial_tp_idx]

            #       0. quantity validation
            #           a. qty should not be zero
            if partial_qty == 0:
                partial_tp_idx += 1
            else:
                #   1. tp_order
                order_no, _, error_code = self.limit_order_v2(order_info, request_name="PARTIAL_LIMIT_CLOSE", code=code, order_side=close_side, qty=partial_qty, price=partial_tp)

                #       a. if limit_order succeed, error_code = 0
                if not error_code:
                    order_no_list[partial_tp_idx] = order_no
                    # order_no_list.append(order_no)
                    partial_tp_idx += 1

                #       b. errors (Todo)
                else:
                    #           i. unknown error : retry
                    #   # <-- 추후 에러처리를 위해 return 시킴
                    # if error_code == "unknown":
                    #     return order_no_list, error_code
                    # #           ii. known error : retry
                    # #                   1. -1111 : quantity precision error
                    # #                   2. -4003 : quantity less than zero
                    # #                   3. -2019 (Margin is insufficient) or something else
                    # else:
                    # time.sleep(self.config.trader_set.order_term)
                    # continue
                    return order_no_list, error_code

        return order_no_list, error_code

    def cancel_order(self, **kwargs):

        """
        취소에 사용되는 qty, price = 1, 0 로 설정해도 무방함.
        """
        price = 0

        order_side = kwargs['order_side']
        if order_side == "BUY":
            order_side_number = 3
        else:
            order_side_number = 4

        while 1:
            try:
                # 2. code essential
                self.send_order("CANCEL", self.screen_number, self.account_number,
                                order_side_number, kwargs['code'], kwargs['qty'], price, "", kwargs["origin_order_no"])

            except Exception as e:
                self.sys_log.error('error in cancel_order : {}'.format(e))
                time.sleep(self.config.trader_set.order_term)
                continue

            else:
                """
                3. cancel_order 의 경우, 미체결 잔량이 없는 상황에서 "" 을 return 하기 때문에 send_order 자체의 error 가 아닌 이상, pass 함.
                    a. 공백이라는 log 만 남기도록 함.
                """
                if self.order_no == "":
                    self.sys_log.error("cancel_order : self.order_no == "", maybe origin_order_no already executed.")
                #     time.sleep(self.config.trader_set.order_term)
                #     continue
                else:
                    self.sys_log.info('{} cancel_order executed.'.format(kwargs["origin_order_no"]))
                # 사실상, return value 를 사용하고 있지 않음.
                return  # self.order_no

    # def cancel_order_list_v2(self, code, order_side, order_info_valid, order_type=OrderType.LIMIT):
    #
    #     """
    #     1. order_info 를 수신해 loc 접근으로만 return data 를 추출한다.
    #     2. open_exec_loop 내부에서 [order_no] 를 사용해 order_no_list 를 유지하는 이유는 partial_tps cancel 에도 사용되기 때문.
    #     """
    #
    #     #     1. order_no_list 의 "" (공백) 에 대한 무결성 검증해야함. (loc 접근할때 안되니까)
    #     order_no_arr = np.array(order_no_list)
    #     order_no_arr = order_no_arr[np.where(order_no_arr != "")[0]]
    #
    #     # ------------ cancel limit_tp orders ------------ #
    #     #      1. post_order_res_list 를 받아서 해당 order 만 취소
    #     #          a. None 포함된 경우는 ? - None 처리 됨
    #     #          b. 이미 완료된 경우 - cancel_order 가능한가 ? nope
    #     if order_type == OrderType.LIMIT:
    #         [self.cancel_order(code=code, order_side=order_side, origin_order_no=order_no)
    #          for order_no in order_info_valid.index]
    #
    #     # -------- 취소한 order 에 대해 체결가, 체결량 조회 -------- #
    #     if order_info is None:
    #         order_info = self.get_order_info(계좌번호=self.account_number, 전체종목구분=0, 매매구분="0", 종목코드="", 체결구분="0")
    #     order_info_valid = order_info.loc[order_no_arr]
    #
    #     #     1. get execPrice_list
    #     exec_price_list = order_info_valid['체결가'].tolist()  # 기존의 list 형태를 유지하기 위해 tolist() 사용함.
    #     #     2. get np.sum(executedQty_list)
    #     exec_qty_list = order_info_valid['체결량'].tolist()
    #     #     3. get origin_qty
    #     orig_qty_list = order_info_valid['주문수량'].tolist()
    #
    #     return exec_price_list, np.sum(exec_qty_list), np.sum(orig_qty_list)

    def cancel_order_list(self, code, order_side, order_no_list_valid, order_info=None, order_type=OrderType.LIMIT):

        """
        0. market_close 에서
        1. order_info 를 수신해 loc 접근으로만 return data 를 추출한다.
        2. open_exec_loop 내부에서 [order_no] 를 사용해 order_no_list_valid 를 유지하는 이유는 partial_tps cancel 에도 사용되기 때문.
            a. 또한 market_close 의 마지막 phase 에서는 order_info 없이 order_no 로만 작업하기 때문에 필요함.
        """

        # 1. order_no_list_valid 의 "" (공백) 에 대한 무결성 검증해야함. (loc 접근할때 안되니까)
        # order_no_arr = np.array(order_no_list_valid)
        # order_no_arr = (order_no_arr[np.where(order_no_arr != "")[0]])

        # 2. post_order_res_list 를 받아서 해당 order 만 취소
        #          a. None 포함된 경우는 ? - None 처리 됨
        #          b. 이미 완료된 경우 - cancel_order 가능한가 ? nope
        #                   i. list comprehension 작동 안함.
        if order_type == OrderType.LIMIT:
            for order_no in order_no_list_valid:
                order_no_str = str(order_no)
                qty = order_info.loc[order_no_str]["미체결수량"]
                self.cancel_order(code=code, order_side=order_side, qty=int(qty), origin_order_no=order_no_str)
                time.sleep(self.config.trader_set.order_term)

        # 3. 취소한 market_close_order 에 대한 체결가, 체결량 조회
        #       a. add term for API limit.
        if order_info is None:
            time.sleep(self.config.trader_set.api_term)
            order_info = self.get_order_info(계좌번호=self.account_number, 전체종목구분=0, 매매구분="0", 종목코드="", 체결구분="0")

        # 4. row 가 1 일 경우, type = series => .loc 으로 조회되지 않는점. : key error
        # if type(order_info) == pd.core.series.Series:
        #     order_info_valid = order_info        #
        #     #           i. series 의 경우 value 가 출력되기 때문에 형태 유지를 위해 list 안에 넣어줌.
        #     exec_price_list = [order_info_valid['체결가']]
        # else:

        order_info_valid = order_info.loc[order_no_list_valid]

        #     a. get execPrice_list
        exec_price_list = order_info_valid['체결가'].tolist()  # 기존의 list 형태를 유지하기 위해 tolist() 사용함.
        #     b. get np.sum(executedQty_list)
        exec_qty_list = order_info_valid['체결량']  # .tolist()
        #     c. get origin_qty : market_close 에서 체결 확인을 위해 origin_qty 를 return 함.
        orig_qty_list = order_info_valid['주문수량']  # .tolist()

        return exec_price_list, np.sum(exec_qty_list), np.sum(orig_qty_list)

    def market_close_order_v3(self, order_no_list_valid, order_info, open_exec_qty, **kwargs):

        """
        v2 -> v3
            1. cancel_order_list_v2 적용으로 인한 order_info parameter 추가
            2. close_remain_quantity 을 "미체결수량" 으로부터 직접 구함
                a. remove open_exec_qty
            3. add open_exec_qty for invalid order_no_list.
        """

        while 1:
            # 0. cancel partial_limit_tp order list.
            if len(order_no_list_valid) != 0:
                self.cancel_order_list(kwargs['code'], kwargs['order_side'], order_no_list_valid, order_info)
                #       a. 일반적으로 entry ~ exit 이 몇 틱정도 소요된다고 가정했을때, order_info 를 close_exec 위에서 갱신하는게 무의미해진다고 봄.
                #       b. 보유 수량 = 주문수량 - 체결량 => 한 거래에 대한 모든 주문량을 close 등록했다고 가정했을때, cancel 후 가능 수량은 미체결수량과 동일해짐.
                # close_remain_quantity = np.sum(order_info.loc[order_no_list_valid]["주문수량"]) - np.sum(np.sum(order_info.loc[order_no_list_valid]["체결량"]))
                close_remain_quantity = np.sum(order_info.loc[order_no_list_valid]["미체결수량"])
            else:
                close_remain_quantity = open_exec_qty

            # 2. get price, volume precision
            #       a. quantity_precision fixed in Stock.
            quantity_precision = 0
            self.sys_log.info('quantity_precision : {}'.format(quantity_precision))

            close_remain_quantity = self.calc_with_precision(close_remain_quantity, quantity_precision)
            self.sys_log.info('close_remain_quantity : {}'.format(close_remain_quantity))

            order_side = kwargs['order_side']
            if order_side == "BUY":
                order_side_number = 1
            else:
                order_side_number = 2

            # error_code = None
            while 1:
                try:
                    self.send_order("MARKET", self.screen_number, self.account_number,
                                    order_side_number, kwargs['code'], int(close_remain_quantity), 0, "03", "")

                except Exception as e:
                    self.sys_log.error('error in market_close_order : {}'.format(e))

                    # 1. error 처리 (Todo)
                    #       a. unknown error : return (retry 가 의미 없을테니까)
                    # return
                    #       b. known error : retry
                    #               i. -2022 ReduceOnly Order is rejected
                    #               ii. Precision is over the maximum defined for this asset
                    time.sleep(self.config.trader_set.order_term)
                    continue

                else:
                    # 2. 여기가 공백인거는 미체결 quantity 가 있다는 건데, 기존 close_order 를 재취소하는게 맞는 것 같음.
                    if self.order_no == "":
                        self.sys_log.error("market_order : self.order_no == "", cancel reorder.")
                        # time.sleep(self.config.trader_set.order_term)
                        # continue
                    else:
                        self.sys_log.info('market_order enlisted. : {}'.format(datetime.now()))
                    break

            if self.order_no == "":
                time.sleep(self.config.trader_set.order_term)
                continue

            # order 수행 후 바로 order_info 조회 => 이런 문제 때문에, 바로 cancel_order_list_v1 type 이 유효한 것.
            market_exec_price_list, market_exec_qty, market_orig_qty = self.cancel_order_list(kwargs['code'], kwargs['order_side'],
                                                                                              [self.order_no], order_type=OrderType.MARKET)

            if market_orig_qty - market_exec_qty < 1:  # Stock 특성상 quantity_precision = 0, devision by zero 발생해서 "< 1" 로 설정함.
                self.sys_log.info('market close order executed.')
                self.sys_log.info("market_exec_price_list : {}".format(market_exec_price_list))
                return market_exec_price_list
            else:
                # 3. reorder
                self.sys_log.info('market_close reorder.')
                time.sleep(self.config.trader_set.order_term)
                continue

    def market_close_order_v2(self, order_no_list, open_exec_qty, **kwargs):

        while 1:  # market reorder 를 위한 loop
            _, tp_exec_qty, _ = self.cancel_order_list(kwargs['code'], kwargs['order_side'], order_no_list)

            # ------------ get close_remain_quantity ------------ #
            close_remain_quantity = open_exec_qty - tp_exec_qty

            # ------------ get price, volume precision ------------ #
            quantity_precision = 0
            self.sys_log.info('quantity_precision : {}'.format(quantity_precision))

            close_remain_quantity = self.calc_with_precision(close_remain_quantity, quantity_precision)
            self.sys_log.info('close_remain_quantity : {}'.format(close_remain_quantity))

            order_side = kwargs['order_side']
            if order_side == "BUY":
                order_side_number = 1
            else:
                order_side_number = 2

            error_code = None
            while 1:
                try:
                    self.send_order("MARKET", self.screen_number, self.account_number,
                                    order_side_number, kwargs['code'], kwargs['qty'], 0, "03", "")

                except Exception as e:
                    self.sys_log.error('error in market_close_order : {}'.format(e))

                    # Todo, error_code 처리
                    # ------------ check error codes ------------ #
                    # ------ -2022 ReduceOnly Order is rejected ------ #
                    # ------ -1111 : Precision is over the maximum defined for this asset ------ #
                    return

                else:
                    # self.sys_log.info('limit_order enlisted. : {}'.format(datetime.now()))
                    break

            market_exec_price_list, market_exec_qty, market_orig_qty = self.cancel_order_list(kwargs['code'], kwargs['order_side'],
                                                                                              [self.order_no], order_type=OrderType.MARKET)

            if market_orig_qty - market_exec_qty < 1:  # Stock 특성상 quantity_precision = 0, devision by zero 발생해서 "< 1" 로 설정함.
                self.sys_log.info('market close order executed')
                self.sys_log.info("market_executedPrice_list : {}".format(market_exec_price_list))
                return market_exec_price_list
            else:
                # ------ complete close by market ------ #
                self.sys_log.info('market_close reorder')
                continue