from funcs.binance.futures_module_v4 import FuturesModule
from binance_f.model import *

from funcs.binance.futures_concat_candlestick_ftr_v2 import concat_candlestick
from funcs.public.broker import calc_rows_and_days, itv_to_number

import numpy as np
import pandas as pd
from datetime import datetime
import time
from ast import literal_eval
import pickle


class ShieldModule(FuturesModule):

    """
    Shield 의 정상적인 동작을 위한 각 phase 의 method 가 모인 Class
    v4 -> v4_1
        1. allow server_time.
    """

    def __init__(self, config):
        api_key, secret_key = self.load_key(r"D:\Projects\System_Trading\JnQ\Bank\api_keys\binance_mademerich.pkl")
        FuturesModule.__init__(self, config, api_key=api_key, secret_key=secret_key, receive_limit_ms=1000*3600)

    @staticmethod
    def load_key(key_abspath):
        with open(key_abspath, 'rb') as f:
            return pickle.load(f)

    def posmode_margin_leverage(self, code):

        while 1:  # for completion
            # 0. position mode => hedge_mode (long & short)
            try:
                server_time = self.time()['serverTime']
                self.change_position_mode(dualSidePosition="true", recvWindow=2000, timestamp=server_time)
            except Exception as e:
                #   i. allow -4059, "No need to change position side."
                if '-4059' not in str(e):
                    msg = "error in change_position_mode : {}".format(e)
                    self.sys_log.error(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
            else:
                self.sys_log.info("position mode --> hedge_mode")

            # 1. margin type => "cross or isolated"
            try:
                self.change_margin_type(symbol=code, marginType=FuturesMarginType.CROSSED, recvWindow=6000, timestamp=server_time)
                # self.change_margin_type(symbol=code, marginType=FuturesMarginType.ISOLATED, recvWindow=6000, timestamp=server_time)
            except Exception as e:
                #   i. allow -4046, "No need to change margin type."
                if '-4046' not in str(e):
                    msg = "error in change_margin_type : {}".format(e)
                    self.sys_log.error(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
            else:
                self.sys_log.info("leverage type --> isolated")

            # 2. check limit leverage
            #       a. limit_leverage == None 인 경우만, server 로 부터 limit_leverage 불러옴.
            if self.config.lvrg_set.limit_leverage == "None":
                try:
                    limit_leverage = self.get_limit_leverage(code)
                except Exception as e:
                    msg = "error in get limit_leverage : {}".format(e)
                    self.sys_log.error(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                    # 'Symbol is closed.'
                    if '-4141' in str(e):
                        return
                    continue
                else:
                    self.sys_log.info('limit_leverage : {}'.format(limit_leverage))
                    return limit_leverage
            else:
                #   b. default = 1.
                limit_leverage = 1
                self.sys_log.info('limit_leverage : {}'.format(limit_leverage))
                return limit_leverage

    def get_open_side_v2(self, res_df, papers, np_timeidx, open_num=1):

        public, utils_list, config_list = papers
        utils, config = None, config_list[0]    # 첫 config 로 초기화.
        open_side = None

        for utils_, cfg_ in zip(utils_list, config_list):

            # 1. point
            #       a. 여기 왜 latest_index 사용했는지 모르겠음 -> bar_close point 를 고려해 mr_res 와의 index 분리
            #       b. vecto. 하는 이상, point x ep_loc 의 index 는 sync. 되어야함 => complete_index 사용 (solved)
            #  Todo, 각 phase 분리한 이유 = sys_logging (solved)
            if res_df['short_open{}_{}'.format(open_num, cfg_.selection_id)].to_numpy()[cfg_.trader_set.complete_index]:
                self.sys_log.warning("[ short ] true_point selection_id : {}".format(cfg_.selection_id))

                # 2. ban
                if not cfg_.pos_set.short_ban:

                    # 3. mr_res
                    if open_num == 1:
                        mr_res, zone_arr = public.ep_loc_p1_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL)
                    else:
                        mr_res, zone_arr = public.ep_loc_p2_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.SELL)
                    self.sys_log.warning("mr_res[cfg_.trader_set.complete_index] : {}\n".format(mr_res[cfg_.trader_set.complete_index]))

                    # 4. assign
                    if mr_res[cfg_.trader_set.complete_index]:
                        utils = utils_
                        config = cfg_
                        open_side = OrderSide.SELL
                        break
                else:
                    self.sys_log.warning("cfg_.pos_set.short_ban : {}".format(cfg_.pos_set.short_ban))

            # 1. point
            #       a. 'if' for nested ep_loc.point        #
            if res_df['long_open{}_{}'.format(open_num, cfg_.selection_id)].to_numpy()[cfg_.trader_set.complete_index]:
                self.sys_log.warning("[ long ] true_point selection_id : {}".format(cfg_.selection_id))

                # 2. ban
                if not cfg_.pos_set.long_ban:  # additional const. on trader

                    # 3. mr_res
                    if open_num == 1:
                        mr_res, zone_arr = public.ep_loc_p1_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY)
                    else:
                        mr_res, zone_arr = public.ep_loc_p2_v3(res_df, cfg_, np_timeidx, show_detail=True, ep_loc_side=OrderSide.BUY)
                    self.sys_log.warning("mr_res[cfg_.trader_set.complete_index] : {}\n".format(mr_res[cfg_.trader_set.complete_index]))

                    # 4. assign
                    if mr_res[cfg_.trader_set.complete_index]:
                        utils = utils_
                        config = cfg_
                        open_side = OrderSide.BUY
                        break
                else:
                    self.sys_log.warning("cfg_.pos_set.long_ban : {}".format(cfg_.pos_set.long_ban))

        return open_side, utils, config

    def get_streamer(self):
        row_list = literal_eval(self.config.trader_set.row_list)
        itv_list = literal_eval(self.config.trader_set.itv_list)
        rec_row_list = literal_eval(self.config.trader_set.rec_row_list)
        use_rows, days = calc_rows_and_days(itv_list, row_list, rec_row_list)

        back_df = pd.read_feather(self.config.trader_set.back_data_path, columns=None, use_threads=True).set_index("index")

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

    def get_new_df_onstream(self, streamer):
        try:
            res_df = next(streamer)  # Todo, 무결성 검증 미진행
        except Exception as e:
            msg = "error in get_new_df_onstream : {}".format(e)
            self.sys_log.error(msg)
            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
            self.kill_proc()  # error 처리할 필요없이, backtrader 프로그램 종료
            # return None, None   # output len 유지
        else:
            return res_df

    def get_new_df(self, code, calc_rows=True, mode="OPEN"):
        while 1:
            try:
                row_list = literal_eval(self.config.trader_set.row_list)
                if calc_rows:
                    itv_list = literal_eval(self.config.trader_set.itv_list)
                    rec_row_list = literal_eval(self.config.trader_set.rec_row_list)
                    use_rows, days = calc_rows_and_days(itv_list, row_list, rec_row_list)
                else:
                    use_rows, days = row_list[0], 1  # load_new_df3 사용 의도 = close, time logging
                # print(use_rows, days)
                use_rows = min(1500, use_rows)

                new_df_, _ = concat_candlestick(code, '1m',
                                                days=days,
                                                limit=use_rows,
                                                timesleep=None,
                                                show_process=0)

                # 1. new_df_ validity check (Stock 의 경우 59.999 초를 더해주어야함, timeindex 의 second = 00)
                #    a. datetime timestamp() 의 기준은 second 임.
                if str(new_df_.index[-1]).split(':')[-1] == '00':
                    last_row_timestamp = datetime.timestamp(new_df_.index[-1]) + 59.999
                else:
                    last_row_timestamp = datetime.timestamp(new_df_.index[-1])

                if last_row_timestamp < datetime.now().timestamp():
                    self.sys_log.warning("dataframe last row validation warning, last index : {}".format(str(new_df_.index[-1])))
                    time.sleep(self.config.trader_set.api_term)
                    continue

                if mode == "OPEN":
                    self.sys_log.info("complete new_df_'s time : {}".format(datetime.now()))

            except Exception as e:
                msg = "error in get_new_df : {}".format(e)
                self.sys_log.error(msg)
                if 'sum_df' not in msg:
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                #        1. adequate term for retries
                time.sleep(self.config.trader_set.api_term)
                continue
            else:
                return new_df_

    def get_tpepout(self, open_side, res_df_open, res_df):

        self.sys_log.info('open_side : {}'.format(open_side))
        selection_id = self.config.selection_id

        tp_j = self.config.trader_set.complete_index
        ep_j = self.config.trader_set.complete_index
        # Todo, complete_index + 1 이 아니라, 해당 index 롤 res_df_open 이 아닌 -> res_df 에 입력하면 되는거아닌가
        #  -> use_point2 여부에 따라 다르게 작성할 필요가 없어짐 (solved)
        out_j = self.config.trader_set.complete_index

        # if selection_id in ['v5_2']:
        # if self.config.ep_set.point2.use_point2:
        #     # out_j = self.config.trader_set.complete_index + 1  # Todo, why + 1 ?
        #     out_j = self.config.trader_set.complete_index
        # else:
        #     out_j = self.config.trader_set.complete_index

        #       Todo        #
        #        b. --> res_df 사용시, dynamic (pre, whole) 을 허용하겠다는 의미
        #           i. ep_loc_point2 미사용시, res_df = res_df_open
        if open_side == OrderSide.BUY:
            ep = res_df_open['long_ep1_{}'.format(selection_id)].to_numpy()[ep_j]

            if self.config.pos_set.long_inversion:
                open_side = OrderSide.SELL  # side change (inversion)
                out = res_df['long_tp_{}'.format(selection_id)].to_numpy()[out_j]
                tp = res_df_open['long_out_{}'.format(selection_id)].to_numpy()[tp_j]
            else:
                out = res_df['long_out_{}'.format(selection_id)].to_numpy()[out_j]
                tp = res_df_open['long_tp_{}'.format(selection_id)].to_numpy()[tp_j]
        else:
            ep = res_df_open['short_ep1_{}'.format(selection_id)].to_numpy()[ep_j]

            if self.config.pos_set.short_inversion:
                open_side = OrderSide.BUY
                out = res_df['short_tp_{}'.format(selection_id)].to_numpy()[out_j]
                tp = res_df_open['short_out_{}'.format(selection_id)].to_numpy()[tp_j]
            else:
                out = res_df['short_out_{}'.format(selection_id)].to_numpy()[out_j]
                tp = res_df_open['short_tp_{}'.format(selection_id)].to_numpy()[tp_j]

        return tp, ep, out, open_side

    def check_open_exec_qty_v2(self, post_order_res, open_quantity):
        """
        1. breakout_gty_ratio
            a. exec_open_qty 와 open_quantity 의 ratio 비교
            b. realtime_term 에 delay 주지 않기 위해, term 설정
        2. 이걸 설정 안해주면 체결되도 close_order 진행 안됨
        """
        # Todo
        #  1. 한번 멈췄는데 이유를 모르겠음, exec_qty, open_qty 둘다 잘찍히는데.. runtime 상의 오류인가?
        #       a. order_info['status'] == "FILLED" 일단은, 추가해봄.
        order_info = self.get_order_info(post_order_res['symbol'], post_order_res['orderId'])
        open_exec = (abs(self.get_exec_qty(order_info) / open_quantity) >= self.config.trader_set.open_exec_qty_ratio) or order_info['status'] == "FILLED"
        return open_exec

    def check_breakout_qty(self, first_exec_qty_check, check_time, post_order_res, open_quantity):
        #        1. breakout_gty_ratio
        #           a. exec_open_qty 와 open_quantity 의 ratio 비교
        #           b. realtime_term 에 delay 주지 않기 위해, term 설정
        #        2. 이걸 설정 안해주면 체결되도 close_order 진행 안됨

        # Todo
        #  1. 한번 멈췄는데 이유를 모르겠음, exec_qty, open_qty 둘다 잘찍히는데.. runtime 상의 오류인가?
        if first_exec_qty_check or time.time() - check_time >= self.config.trader_set.qty_check_term:
            order_info = self.get_order_info(post_order_res['symbol'], post_order_res['orderId'])
            breakout = abs(self.get_exec_qty(order_info) / open_quantity) >= self.config.trader_set.breakout_qty_ratio
            return 0, time.time(), breakout
        return 0, check_time, 0  # breakout_qty survey 실행한 경우만 check_time 갱신

    def check_ei_k_v3(self, res_df_open, res_df, realtime_price, open_side, expired):

        """
        v2 --> v3
            1. expired variable added.
        """

        selection_id = self.config.selection_id

        if self.config.tr_set.expire_tick != "None":
            if time.time() - datetime.timestamp(res_df_open.index[self.config.trader_set.latest_index]) \
                    >= self.config.tr_set.expire_tick * 60:
                expired = 1

        # Todo, expire_k2 vs expire_k1, 추후 p2 사용시 재고해야할 것
        if self.config.tr_set.expire_k1 != "None":  # Todo -> 이거 None 이면 "무한 대기" 발생 가능함

            # realtime_price = self.get_market_price_v3()

            tp_j = self.config.trader_set.complete_index
            #       Todo        #
            #        2. 추후, dynamic_tp 사용시 res_df 갱신해야할 것
            #           a. 그에 따른 res_df 종속 변수 check
            #        3. ei_k - expired 변수 달아주고, close bar waiting 추가할지 고민중
            #        4. warning - 체결량 존재하는데, expired 가능함 - market 고민중
            #        5. funcs_trader_modules 에서 order_side check 하는 function 모두 inversion 고려해야할 것
            #           a. "단, order_side change 후로만 해당됨
            if open_side == OrderSide.SELL:
                short_tp_ = res_df_open['short_tp_{}'.format(selection_id)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
                short_tp_gap_ = res_df_open['short_tp_gap_{}'.format(selection_id)].to_numpy()
                if realtime_price <= short_tp_[tp_j] + short_tp_gap_[tp_j] * self.config.tr_set.expire_k1:
                    expired = 1
            else:
                long_tp_ = res_df_open['long_tp_{}'.format(selection_id)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
                long_tp_gap_ = res_df_open['long_tp_gap_{}'.format(selection_id)].to_numpy()
                if realtime_price >= long_tp_[tp_j] - long_tp_gap_[tp_j] * self.config.tr_set.expire_k1:
                    expired = 1

        if expired:
            self.sys_log.warning("cancel open_order by ei_k\n")

        return expired

    def check_ei_k_v2(self, res_df_open, res_df, realtime_price, open_side):

        """
        = expire_tp
        """

        ep_out = 0
        selection_id = self.config.selection_id

        if self.config.tr_set.expire_tick != "None":
            if time.time() - datetime.timestamp(res_df_open.index[self.config.trader_set.latest_index]) \
                    >= self.config.tr_set.expire_tick * 60:
                ep_out = 1

        # Todo, expire_k2 vs expire_k1, 추후 p2 사용시 재고해야할 것
        if self.config.tr_set.expire_k1 != "None":  # Todo -> 이거 None 이면 "무한 대기" 발생 가능함

            # realtime_price = self.get_market_price_v3()

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

    def check_ei_k_onbarclose_v3(self, res_df_open, res_df, e_j, tp_j, open_side, expired):  # for point2

        """
        = expire_tp
        """

        selection_id = self.config.selection_id

        if self.config.tr_set.expire_tick != "None":
            if datetime.timestamp(res_df.index[-1]) - datetime.timestamp(res_df_open.index[self.config.trader_set.latest_index]) \
                    >= self.config.tr_set.expire_tick * 60:
                expired = 1

        if self.config.tr_set.expire_k1 != "None":  # Todo - onbarclose 에서는, res_df_open 으로 open_index 의 tp 정보를 사용
            if open_side == OrderSide.SELL:
                low = res_df['low'].to_numpy()
                short_tp_ = res_df_open['short_tp_{}'.format(selection_id)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
                short_tp_gap_ = res_df_open['short_tp_gap_{}'.format(selection_id)].to_numpy()
                if low[e_j] <= short_tp_[tp_j] + short_tp_gap_[tp_j] * self.config.tr_set.expire_k1:
                    expired = 1
            else:
                high = res_df['high'].to_numpy()
                long_tp_ = res_df_open['long_tp_{}'.format(selection_id)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
                long_tp_gap_ = res_df_open['long_tp_gap_{}'.format(selection_id)].to_numpy()
                if high[e_j] >= long_tp_[tp_j] - long_tp_gap_[tp_j] * self.config.tr_set.expire_k1:
                    expired = 1

        if expired:
            self.sys_log.warning("cancel open_order by ei_k\n")

        return expired

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

    def get_partial_tp_qty(self, ep, tp, open_exec_qty, price_precision, quantity_precision, close_side):
        partial_ranges, partial_qty_ratio = literal_eval(self.config.tp_set.partial_ranges), literal_eval(self.config.tp_set.partial_qty_ratio)
        partial_length = len(partial_ranges)
        assert partial_length == len(partial_qty_ratio)
        en_ps, tps, open_qtys = [np.tile(arr_, (partial_length,)) for arr_ in [ep, tp, open_exec_qty]]

        if close_side == OrderSide.BUY:  # short
            partial_tps = en_ps - (en_ps - tps) * partial_ranges
        else:  # long
            partial_tps = en_ps + (tps - en_ps) * partial_ranges
        partial_qtys = open_qtys * partial_qty_ratio

        partial_tps = list(map(lambda x: self.calc_with_precision(x, price_precision), partial_tps))
        partial_qtys = list(map(lambda x: self.calc_with_precision(x, quantity_precision), partial_qtys))
        self.sys_log.info('partial_tps : {}'.format(partial_tps))
        self.sys_log.info('partial_qtys : {}'.format(partial_qtys))

        return partial_tps, partial_qtys

    def check_limit_tp_exec_v2(self, post_order_res_list, quantity_precision):

        """
        v1 -> v2
            1. tp_type 에 의한 condition phase 를 제거함.
                a. 외부에서 참조하도록 구성함.
            2. return_price parameter 제거하고 all_executed, exec_price_list 반환
            3. add len(order_exec_check_list) validation.
        """

        all_executed = 0
        order_info_list = [self.get_order_info(post_order_res['symbol'], post_order_res['orderId'])
                           for post_order_res in post_order_res_list if post_order_res is not None]
        order_exec_check_list = [self.check_execution(order_info, quantity_precision) for order_info in order_info_list]

        # 손매매로 인한 order_info['status'] == EXPIRED 는 고려 대상이 아님. (손매매 이왕이면 하지 말라는 이야기)
        order_exec_check_list_len = len(order_exec_check_list)
        if order_exec_check_list_len != 0:
            if np.sum(order_exec_check_list) == order_exec_check_list_len:
                self.sys_log.info("all limit tp order executed")
                all_executed = 1

        exec_price_list = [self.get_exec_price(order_info) for order_info, executed in zip(order_info_list, order_exec_check_list) if executed]
        return all_executed, exec_price_list

    def check_hl_out_onbarclose_v2(self, res_df, market_close_on, out, liqd_p, open_side):

        """
        v1 -> v2
            1. add liquidation platform.
            2. add log_out
            3. add non_out
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

        if not self.config.out_set.non_out:

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
            4. add non_out.
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

        if not self.config.out_set.non_out:

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

    def check_signal_out_v4(self, res_df, np_timeidx, market_close_on, open_side):

        """
        v3 -> v4
            1. add fisher_exit.
            2. add np_timeidx
        """

        log_out = None
        c_i = self.config.trader_set.complete_index

        close = res_df['close'].to_numpy()[c_i]

        # 1. timestamp
        # if self.config.out_set.tf_exit != "None":
        #     if np_timeidx[i] % self.config.out_set.tf_exit == self.config.out_set.tf_exit - 1 and i != open_i:
        #         market_close_on = True

        # 2. fisher
        if self.config.out_set.fisher_exit:

            itv_num = itv_to_number(self.config.loc_set.point1.tf_entry)

            if np_timeidx[c_i] % itv_num == itv_num - 1:

                fisher_ = res_df['fisher_{}30'.format(self.config.loc_set.point1.tf_entry)].to_numpy()
                fisher_band = self.config.out_set.fisher_band
                fisher_band2 = self.config.out_set.fisher_band2

                if open_side == OrderSide.SELL:
                    if (fisher_[c_i - itv_num] > -fisher_band) & (fisher_[c_i] <= -fisher_band):
                        market_close_on = True
                    elif (fisher_[c_i - itv_num] < fisher_band2) & (fisher_[c_i] >= fisher_band2):
                        market_close_on = True
                else:
                    if (fisher_[c_i - itv_num] < fisher_band) & (fisher_[c_i] >= fisher_band):
                        market_close_on = True
                    elif (fisher_[c_i - itv_num] > fisher_band2) & (fisher_[c_i] <= fisher_band2):
                        market_close_on = True

        # 3. rsi_exit
        if self.config.out_set.rsi_exit:
            rsi_ = res_df['rsi_%s' % self.config.loc_set.point.exp_itv].to_numpy()
            osc_band = self.config.loc_set.point.osc_band

            if open_side == OrderSide.SELL:
                if (rsi_[c_i - 1] >= 50 - osc_band) & (rsi_[c_i] < 50 - osc_band):
                    market_close_on = True
            else:
                if (rsi_[c_i - 1] <= 50 + osc_band) & (rsi_[c_i] > 50 + osc_band):
                    market_close_on = True

        # 4. cci_exit
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

    def check_signal_out_v3(self, res_df, market_close_on, open_side):

        """
        v3 -> v4
            1. add fisher_exit.
            2. add np_timeidx
        """

        log_out = None
        c_i = self.config.trader_set.complete_index

        close = res_df['close'].to_numpy()[c_i]

        # 1. timestamp
        # if self.config.out_set.tf_exit != "None":
        #     if np_timeidx[i] % self.config.out_set.tf_exit == self.config.out_set.tf_exit - 1 and i != open_i:
        #         market_close_on = True

        # 3. rsi_exit
        if self.config.out_set.rsi_exit:
            rsi_ = res_df['rsi_%s' % self.config.loc_set.point.exp_itv].to_numpy()
            osc_band = self.config.loc_set.point.osc_band

            if open_side == OrderSide.SELL:
                if (rsi_[c_i - 1] >= 50 - osc_band) & (rsi_[c_i] < 50 - osc_band):
                    market_close_on = True
            else:
                if (rsi_[c_i - 1] <= 50 + osc_band) & (rsi_[c_i] > 50 + osc_band):
                    market_close_on = True

        # 4. cci_exit
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

    def limit_order_v2(self, code, order_type, limit_side, pos_side, limit_price, limit_quantity, order_data=None):

        """
        _ -> v2
            1. code 도입.
            2. rename to retry_count & -1111 error update.
        """
        retry_count = 0

        if order_data is None:
            over_balance = None
        else:
            over_balance, leverage = order_data

        limit_quantity_org = limit_quantity

        while 1:
            try:
                # 0. new_order (limit & market)
                # Todo, quantity 에 float 대신 str 을 입력하는게 정확한 것으로 앎.
                #   1. precision 보호 차원인가.
                server_time = self.time()['serverTime']
                if order_type == OrderType.MARKET:
                    # a. market order
                    post_order_res = self.new_order(symbol=code,
                                                    side=limit_side,
                                                    positionSide=pos_side,
                                                    type=OrderType.MARKET,
                                                    quantity=str(limit_quantity),
                                                    timestamp=server_time)
                else:
                    # b. limit order
                    post_order_res = self.new_order(timeInForce=TimeInForce.GTC,
                                                    symbol=code,
                                                    side=limit_side,
                                                    positionSide=pos_side,
                                                    type=OrderType.LIMIT,
                                                    quantity=str(limit_quantity),
                                                    price=str(limit_price),
                                                    timestamp=server_time)
            except Exception as e:
                msg = "error in limit_order : {}".format(e)
                self.sys_log.error(msg)
                self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                # 1. limit_order() 에서 해결할 수 없는 error 일 경우, return
                #       a. -4003 : quantity less than zero
                if "-4003" in str(e):
                    error_code = -4003
                    return None, over_balance, error_code

                # 2. limit_order() 에서 해결할 error 일 경우, continue
                #       y. -1111 : Precision is over the maximum defined for this asset = quantity precision error
                if '-1111' in str(e):

                    try:
                        #   _, quantity_precision = self.get_precision()
                        quantity_precision = retry_count

                        # close_remain_quantity = get_remaining_quantity(code)
                        #   error 발생한 경우 정상 체결된 qty 는 존재하지 않는다고 가정 - close_remain_quantity 에는 변함이 없을거라 봄
                        limit_quantity = self.calc_with_precision(limit_quantity_org, quantity_precision, def_type='floor')
                        self.sys_log.info("modified qty & precision : {} {}".format(limit_quantity, quantity_precision))

                    except Exception as e:
                        msg = "error in get modified qty_precision : {}".format(e)
                        self.sys_log.error(msg)
                        self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                    retry_count += 1
                    if retry_count >= 10:
                        retry_count = 0

                    # error_code = -1111
                    # return None, over_balance, error_code
                    time.sleep(self.config.trader_set.order_term)
                    continue

                #       a. -4014 : price precision error
                if '-4014' in str(e):
                    try:
                        realtime_price = self.get_market_price_v3(code)
                        price_precision = self.get_precision_by_price(realtime_price)
                        limit_price = self.calc_with_precision(limit_price, price_precision)
                        self.sys_log.info("modified price & precision : {}, {}".format(limit_price, price_precision))

                    except Exception as e:
                        msg = "error in get_market_price_v3 (open_order phase): {}".format(e)
                        self.sys_log.error(msg)
                        self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                    time.sleep(self.config.trader_set.order_term)
                    continue

                #       b. -2019 : Margin is insufficient
                # Todo - for tp_exec_qty miscalc., self.overbalance 수정되었으면, return 으로 받아야할 것 => 무슨 소리야..?
                if '-2019' in str(e):
                    if order_data is not None:  # None 인 경우, leverage 가 정의되지 않음.
                        # i. 예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치
                        try:
                            # 1. get available quantity
                            available_balance = self.get_available_balance() * 0.9

                            _, quantity_precision = self.get_precision(code)
                            quantity = available_balance / limit_price * leverage
                            quantity = self.calc_with_precision(quantity, quantity_precision, def_type='floor')
                            self.sys_log.info('available_balance (temp) : {}'.format(available_balance))
                            self.sys_log.info('quantity : {}'.format(quantity))

                        except Exception as e:
                            msg = "error in get_availableBalance (-2019 phase) : {}".format(e)
                            self.sys_log.error(msg)
                            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                        # ii. balance 가 근본적으로 부족한 경우, 주문을 허용하지 않는다.
                        retry_count += 1
                        if retry_count > 5:
                            msg = "retry_count over. : {}".format(retry_count)
                            self.sys_log.error(msg)
                            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                            error_code = -2019
                            return None, over_balance, error_code
                        else:
                            time.sleep(self.config.trader_set.order_term)
                            continue
                else:
                    msg = "unknown error : {}".format(e)
                    self.sys_log.error(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                    error_code = 'unknown'
                    return None, over_balance, error_code

            else:
                # 3. 정상 order 의 error_code = 0.
                self.sys_log.info("limit_order enlisted. : {}".format(datetime.now()))
                return post_order_res, over_balance, 0

    def partial_limit_order_v5(self, code, post_order_res_list, partial_tps, partial_qtys, close_side, pos_side, open_exec_qty, quantity_precision):

        """
        v4 -> v5
            1. Stock 의 partial_limit_order 구조를 가져옴.
            2. "" --> None.
        """

        partial_tp_idx = 0
        partial_length = len(partial_tps)
        # post_order_res_list = []
        while 1:
            if partial_tp_idx >= partial_length:
                break

            #       x. valid partial_order rejection.
            if post_order_res_list[partial_tp_idx] is not None:
                partial_tp_idx += 1
                continue

            partial_tp = partial_tps[partial_tp_idx]
            #       y. 맨 마지막 잔여 partial_qty => [:-1] indexing 을 사용해 계산함.
            if partial_tp_idx == partial_length - 1:
                partial_qty = self.calc_with_precision(open_exec_qty - np.sum(partial_qtys[:-1]), quantity_precision)
            else:
                partial_qty = partial_qtys[partial_tp_idx]

            # 1. tp_order
            # try:
            """
            1. 들고 있는 position != order_side 일 경우,  -2022,"msg":"ReduceOnly Order is rejected." 발생함
                a. reduceOnly=False 줘도 무관한가 => 를 줘야 무방해짐 (solved)
            """
            post_order_res, _, error_code = self.limit_order_v2(code, OrderType.LIMIT, close_side, pos_side, partial_tp, partial_qty)

            #       a. if limit_order succeed, error_code = 0
            #       b. errors 처리는 limit_order 내부에서 완료하기.
            post_order_res_list[partial_tp_idx] = post_order_res
            partial_tp_idx += 1

        return post_order_res_list

    def cancel_order_wrapper(self, code, order_id):
        while 1:
            try:
                server_time = self.time()['serverTime']
                _ = self.cancel_order(symbol=code, orderId=order_id, timestamp=server_time)
            except Exception as e:
                if '-2011' in str(e):  # -2011 : "Unknown order sent." --> already canceled or filled
                    return
                else:  # = 비정상 error
                    msg = "error in cancel_order : {}".format(e)
                    self.sys_log.error(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
                    time.sleep(self.config.trader_set.order_term)
                    continue
            else:
                return

    def cancel_order_list(self, post_order_res_list, order_type=OrderType.LIMIT):
        """
        1. post_order_res_list 를 받아서 해당 order 만 취소
        2. None 포함된 경우는 ? - None 처리 됨
        3. 이미 완료된 경우 - cancel_order 가능한가 ? nope
        """

        # 1. cancel limit_tp orders
        if order_type == OrderType.LIMIT:
            [self.cancel_order_wrapper(post_order_res['symbol'], post_order_res['orderId']) for post_order_res in post_order_res_list if post_order_res is not None]

        order_info_list = [self.get_order_info(post_order_res['symbol'], post_order_res['orderId'])
                           for post_order_res in post_order_res_list if post_order_res is not None]
        # 2. get execPrice_list
        exec_price_list = [self.get_exec_price(order_info) for order_info in order_info_list]
        # 3. get np.sum(executedQty_list)
        exec_qty_list = [self.get_exec_qty(order_info) for order_info in order_info_list]

        return exec_price_list, np.sum(exec_qty_list)

    def market_close_order_v2(self, code, post_order_res_list, close_side, pos_side, open_exec_qty):

        """
        retry_count updated.
        """

        # 0.This loop exist for complete close
        while 1:
            _, tp_exec_qty = self.cancel_order_list(post_order_res_list)

            # 1. get close_remain_quantity
            close_remain_quantity = open_exec_qty - tp_exec_qty

            # 2. get price, volume precision
            #       a. reatlime 가격 변동으로 인한 precision 변동 가능성 고려
            _, quantity_precision = self.get_precision(code)
            self.sys_log.info('quantity_precision : {}'.format(quantity_precision))

            close_remain_quantity = self.calc_with_precision(close_remain_quantity, quantity_precision)
            self.sys_log.info('close_remain_quantity : {}'.format(close_remain_quantity))

            retry_count = 0
            error_code = 0
            close_remain_quantity_org = close_remain_quantity
            while 1:
                try:
                    server_time = self.time()['serverTime']
                    post_order_res = self.new_order(symbol=code,
                                                    side=close_side,
                                                    positionSide=pos_side,
                                                    type=OrderType.MARKET,
                                                    quantity=str(close_remain_quantity),
                                                    timestamp=server_time)
                except Exception as e:

                    # 3. errors
                    #       a. -2022 ReduceOnly Order is rejected
                    if '-2022' in str(e):
                        error_code = '-2022'
                        break

                    #       b. -4003 quantity less than zero
                    if '-4003' in str(e):
                        error_code = '-4003'
                        break

                    # -2022, -4003 은 logging 하지않는다.
                    msg = "error in close order : {}".format(e)
                    self.sys_log.error(msg)
                    self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                    #       c. -1111 : Precision is over the maximum defined for this asset
                    #           = quantity precision error
                    #           상당히 오래전에 만든 대응법인데, 이게 맞는 걸까.. (Todo)
                    if '-1111' in str(e):

                        try:
                            #   _, quantity_precision = self.get_precision()
                            quantity_precision = retry_count

                            # close_remain_quantity = get_remaining_quantity(code)
                            #   error 발생한 경우 정상 체결된 qty 는 존재하지 않는다고 가정 - close_remain_quantity 에는 변함이 없을거라 봄
                            close_remain_quantity = self.calc_with_precision(close_remain_quantity_org, quantity_precision, def_type='floor')
                            self.sys_log.info("modified qty & precision : {} {}".format(close_remain_quantity, quantity_precision))

                        except Exception as e:
                            msg = "error in get modified qty_precision : {}".format(e)
                            self.sys_log.error(msg)
                            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)

                        retry_count += 1
                        if retry_count >= 10:
                            retry_count = 0

                    # self.config.out_set.out_type = OrderType.MARKET  # just 재시도
                    time.sleep(self.config.trader_set.order_term)
                    continue

                else:
                    self.sys_log.info("market close order enlisted")
                    break

            # 4. term for qty consumption
            time.sleep(1)

            # 5.  post_order_res 가 정의되는 경우 / 그렇지 않은 경우.
            if error_code in ['-2022', '-4003']:
                self.sys_log.info("market close order executed")
                return
            else:
                #   Todo - price_list 제대로 출력하려면, order_type 맞게 명시해야함 (limit 사용 안할거니까, solved)
                market_exec_price_list, market_exec_qty = self.cancel_order_list([post_order_res], order_type=OrderType.MARKET)
                if float(post_order_res['origQty']) - market_exec_qty < 1 / (10 ** quantity_precision):
                    self.sys_log.info("market close order executed")
                    self.sys_log.info("market_executedPrice_list : {}".format(market_exec_price_list))
                    return market_exec_price_list
                else:
                    # ------ complete close by market ------ #
                    self.sys_log.info("market_close reorder")
                    # time.sleep(self.config.trader_set.order_term)   # 위에 time.sleep(1)로 대체.
                    continue
