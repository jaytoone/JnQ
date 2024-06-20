import os

pkg_path = os.path.abspath('./../')
print("pkg_path :", pkg_path)

os.chdir(pkg_path)
# print("os.getcwd() :", os.getcwd())
# quit()

from funcs.binance.futures_concat_candlestick import concat_candlestick
from funcs.binance.order import *
from funcs.public.broker import intmin
from easydict import EasyDict
import numpy as np  # for np.nan
from datetime import datetime
import time
import pickle
from pathlib import Path

#       Todo        #
#        1. check upper dir
#        2. check utils


class Trader:

    def __init__(self, utils_lib, config_name):

        #           static platform variables        #
        self.utils = utils_lib
        self.config_name = config_name

        self.over_balance = None
        self.min_balance = 5.0  # USDT

        self.accumulated_income = 0.0
        self.calc_accumulated_profit = 1.0
        self.accumulated_profit = 1.0
        self.accumulated_back_profit = 1.0

        self.sub_client = None

    def run(self):

        logger_name = "/%s.pkl" % str(datetime.now().timestamp()).split(".")[0]
        trade_log = {}

        exec_file_name = Path(__file__).stem

        print('# ----------- %s ----------- #' % exec_file_name)

        trade_log_path = os.path.join(pkg_path, "trade_log", exec_file_name)
        df_log_path = os.path.join(pkg_path, "df_log", exec_file_name)

        try:
            os.makedirs(trade_log_path, exist_ok=True)
            os.makedirs(df_log_path, exist_ok=True)

        except Exception as e:
            print("error in makedirs for logger :", e)

        config_path = '/config/' + self.config_name

        initial_run = 1
        while 1:

            #       load config (refreshed by every trade)       #
            with open(pkg_path + config_path, 'r') as cfg:
                config = EasyDict(json.load(cfg))

            if config.init_set.symbol_changed:
                initial_run = 1

                #       1. rewrite symbol_changed to false
                config.init_set.symbol_changed = 0

                with open(pkg_path + config_path, 'w') as cfg:
                    json.dump(config, cfg, indent=2)

            if initial_run:

                while 1:    # for completion
                    # --- 1. leverage type => "isolated" --- #
                    try:
                        request_client.change_margin_type(symbol=config.init_set.symbol, marginType=FuturesMarginType.ISOLATED)
                    except Exception as e:
                        print('error in change_margin_type :', e)
                    else:
                        print('leverage type --> isolated')

                    # --- 2. confirm limit leverage --- #
                    try:
                        limit_leverage = get_limit_leverage(symbol_=config.init_set.symbol)
                    except Exception as e:
                        print('error in get limit_leverage :', e)
                        continue
                    else:
                        print('limit_leverage :', limit_leverage)

                    # --- 3. sub_client --- #
                    try:
                        sub_client.subscribe_aggregate_trade_event(config.init_set.symbol.lower(), callback, error)
                        self.sub_client = sub_client
                    except Exception as e:
                        print('error in get sub_client :', e)
                        continue

                    break

                initial_run = 0
                print()

            #       check run        #
            if not config.init_set.run:
                time.sleep(config.init_set.realtime_term)   # give enough time to read config
                continue

            #       init order side      #
            open_side = None

            #       get startTime     #
            start_timestamp = int(time.time() * 1000)

            #       init param.      #
            ep_loc_check = 1
            load_new_df = 1

            while 1:

                if ep_loc_check:

                    #       log last trading time     #
                    #       미체결을 고려해, ep_loc_check 마다 log 수행       #
                    trade_log["last_trading_time"] = str(datetime.now())
                    with open(trade_log_path + logger_name, "wb") as dict_f:
                        pickle.dump(trade_log, dict_f)

                    temp_time = time.time()

                    if load_new_df:

                        try:

                            res_df_list = []
                            old_df = 0
                            for itv_i, interval_ in enumerate(config.init_set.interval_list):

                                if interval_ != "None":

                                    new_df_, _ = concat_candlestick(config.init_set.symbol, interval_,
                                                                    days=1,
                                                                    limit=config.init_set.row_list[itv_i],
                                                                    timesleep=0.2,
                                                                    show_process=False)
                                    if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                        old_df = 1
                                        break
                                    else:
                                        new_df = new_df_.iloc[-config.init_set.row_list[itv_i]:].copy()
                                else:
                                    new_df = None

                                res_df_list.append(new_df)

                            if old_df:
                                continue

                            print("complete load_df execution time :", datetime.now())

                        except Exception as e:
                            print("error in load new_dfs :", e)
                            time.sleep(config.init_set.api_retry_term)
                            continue

                        else:
                            load_new_df = 0

                    print("~ load_new_df time : %.2f" % (time.time() - temp_time))
                    # temp_time = time.time()

                    # ------------ get indi.  ------------ #
                    try:
                        res_df = self.utils.sync_check(res_df_list)  # function usage format maintenance
                        print('~ sync_check time : %.5f' % (time.time() - temp_time))

                    except Exception as e:
                        print("error in sync_check :", e)
                        continue

                    # ------------ enlist_rtc ------------ #
                    try:
                        res_df = self.utils.enlist_rtc(res_df, config)
                        print('~ enlist_rtc time : %.5f' % (time.time() - temp_time))

                    except Exception as e:
                        print("error in enlist_rtc :", e)
                        continue

                    # ------------ enlist_tr ------------ #
                    try:
                        np_timeidx = np.array(list(map(lambda x: intmin(x), res_df.index)))
                        res_df = self.utils.enlist_tr(res_df, config, np_timeidx)
                        print('res_df.index[-1] :', res_df.index[-1])
                        print('~ ep enlist time : %.5f' % (time.time() - temp_time))
                        print()

                    except Exception as e:
                        print('error in enlist_tr :', e)
                        continue

                    # ---------------- ep_loc ---------------- #
                    try:

                        #        save res_df before, entry const check      #
                        if config.init_set.df_log:
                            excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                            # res_df.to_excel(df_log_path + "/%s.xlsx" % excel_name)
                            res_df.reset_index().to_feather(df_log_path + "/%s.ftr" % excel_name, compression='lz4')

                        #           short banned        #
                        # if res_df['entry'][config.init_set.last_index] == config.ep_set.short_entry_score:
                        #     res_df, open_side, _ = self.utils.short_ep_loc(res_df, config, config.init_set.last_index)

                        if res_df['entry'][config.init_set.last_index] == -config.ep_set.short_entry_score:
                            res_df, open_side, _ = self.utils.long_ep_loc(res_df, config, config.init_set.last_index)

                    except Exception as e:
                        print("error in ep_loc phase :", e)
                        continue

                    ep_loc_check = 0   # once used, turn it off

                check_entry_sec = datetime.now().second
                if open_side is not None:

                    #        1. const. open market second
                    if config.ep_set.entry_type == "MARKET":
                        if check_entry_sec > config.ep_set.check_entry_sec:
                            open_side = None
                            print("check_entry_sec :", check_entry_sec)
                            print()
                            continue    # ep_loc_check = False 라서 open_side None phase 로 감

                    #        2. init fee        #
                    if config.ep_set.entry_type == "MARKET":
                        fee = config.init_set.market_fee
                    else:
                        fee = config.init_set.limit_fee

                    break

                    #   1. 추후 position change platform 으로 변경 가능
                    #    a. first_iter 사용

                # ------- open_side is None [ waiting zone ] ------- #
                else:

                    #       check tick change      #
                    #       latest df confirmation       #
                    if datetime.timestamp(res_df.index[-1]) < datetime.now().timestamp():
                        ep_loc_check = 1
                        load_new_df = 1
                        print('res_df[-1] timestamp :', datetime.timestamp(res_df.index[-1]))
                        print('current timestamp :', datetime.now().timestamp())
                        print()

                        #       get configure every minutes      #
                        with open(pkg_path + config_path, 'r') as cfg:
                            config = EasyDict(json.load(cfg))

                    else:
                        time.sleep(config.init_set.realtime_term)  # <-- term for realtime market function

                    # time.sleep(config.init_set.close_complete_term)   # <-- term for close completion

            first_iter = True  # 포지션 변경하는 경우, 필요함
            while 1:  # <-- loop for 'check order type change condition'

                print('open_side :', open_side)

                # ------------- tr setting ------------- #
                # if config.ep_set.entry_type == OrderType.MARKET:
                #     #             ep by realtime_price            #
                #     try:
                #         ep = get_market_price_v2(self.sub_client)
                #     except Exception as e:
                #         print('error in get_market_price :', e)
                #         continue
                # else:

                #             limit & market's ep             #
                #             inversion adjusted
                if open_side == OrderSide.BUY:
                    ep = res_df['long_ep'].iloc[config.init_set.last_index]

                    if config.ep_set.long_inversion:
                        open_side = OrderSide.SELL  # side change (inversion)
                        out = res_df['long_tp'].iloc[config.init_set.last_index]
                        tp = res_df['long_out'].iloc[config.init_set.last_index]
                    else:
                        out = res_df['long_out'].iloc[config.init_set.last_index]
                        tp = res_df['long_tp'].iloc[config.init_set.last_index]
                else:
                    ep = res_df['short_ep'].iloc[config.init_set.last_index]

                    if config.ep_set.short_inversion:
                        open_side = OrderSide.BUY
                        out = res_df['short_tp'].iloc[config.init_set.last_index]
                        tp = res_df['short_out'].iloc[config.init_set.last_index]
                    else:
                        out = res_df['short_out'].iloc[config.init_set.last_index]
                        tp = res_df['short_tp'].iloc[config.init_set.last_index]

                #         get price, volume precision --> reatlime 가격 변동으로 인한 precision 변동 가능성       #
                try:
                    price_precision, quantity_precision = get_precision(config.init_set.symbol)
                except Exception as e:
                    print('error in get price & volume precision :', e)
                    time.sleep(config.init_set.api_retry_term)
                    continue
                else:
                    print('price_precision :', price_precision)
                    print('quantity_precision :', quantity_precision)

                ep = calc_with_precision(ep, price_precision)
                out = calc_with_precision(out, price_precision)  # this means static_out
                tp = calc_with_precision(tp, price_precision)   # this means static_tp
                leverage = self.utils.lvrg_set(res_df, config, limit_leverage, open_side, fee)

                print('ep :', ep)
                print('out :', out)
                print('tp :', tp)
                print('leverage :', leverage)
                print('~ ep out lvrg set time : %.5f' % (time.time() - temp_time))

                try:
                    request_client.change_initial_leverage(symbol=config.init_set.symbol, leverage=leverage)
                except Exception as e:
                    print('error in change_initial_leverage :', e)
                    time.sleep(config.init_set.api_retry_term)
                    continue  # -->  ep market 인 경우에 조심해야함
                else:
                    print('leverage changed -->', leverage)

                if first_iter:
                    # ---------- define start asset ---------- #
                    if self.accumulated_income == 0.0:

                        available_balance = config.init_set.initial_asset  # USDT

                    else:
                        if config.init_set.asset_changed:
                            available_balance = config.init_set.initial_asset

                            #       asset_changed -> 0      #
                            config.init_set.asset_changed = 0

                            with open(pkg_path + config_path, 'w') as cfg:
                                json.dump(config, cfg, indent=2)
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
                        print('available_balance %.3f < min_balance' % available_balance)
                        print()
                        break

                except Exception as e:
                    print('error in get_availableBalance :', e)
                    print()
                    time.sleep(config.init_set.api_retry_term)
                    continue

                print('~ get balance time : %.5f' % (time.time() - temp_time))

                # ---------- get available open_quantity ---------- #
                open_quantity = available_balance / ep * leverage
                open_quantity = calc_with_precision(open_quantity, quantity_precision)
                if self.over_balance is not None:
                    print('available_balance (temp) :', available_balance)
                else:
                    print('available_balance :', available_balance)

                print("open_quantity :", open_quantity)

                real_balance = ep * open_quantity
                print("real_balance :", real_balance)

                # ---------- open order ---------- #
                orderside_changed = False
                self.over_balance, code_1111 = limit_order(self, config.ep_set.entry_type, config, open_side, ep, open_quantity)

                if code_1111:
                    break

                # -------- execution wait time -------- #
                if config.ep_set.entry_type == OrderType.MARKET:

                    #           market : prevent close at open bar           #
                    #           + enough time for open_quantity consumed
                    while 1:
                        if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                            break
                        else:
                            time.sleep(config.init_set.realtime_term)  # <-- for realtime price function

                else:
                    #       limit's ep_out const. & check breakout_qty_ratio         #
                    first_exec_qty_check = 1
                    check_time = time.time()
                    while 1:

                        #       temporary - wait bar close       #
                        # if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):

                        #       check tp_done by hl     #
                        try:
                            realtime_price = get_market_price_v2(self.sub_client)

                        except Exception as e:
                            print('error in get_market_price_v2 (ep_out phase) :', e)
                            continue

                        #        1. tp_j 를 last_index 로 설정해도 갱신되지 않음 (res_df 가 갱신되지 않는한)
                        #       Todo        #
                        #        2. 추후, dynamic_tp 사용시 res_df 갱신해야할 것
                        #           a. 그에 따른 res_df 종속 변수 check
                        tp_j = config.init_set.last_index

                        if open_side == OrderSide.SELL:
                            if realtime_price <= res_df['h_short_rtc_1'].iloc[tp_j] - \
                                    res_df['h_short_rtc_gap'].iloc[tp_j] * config.loc_set.ei_k:
                                print("cancel open_order by ei_k")
                                break
                        else:
                            if realtime_price >= res_df['h_long_rtc_1'].iloc[tp_j] + \
                                    res_df['h_long_rtc_gap'].iloc[tp_j] * config.loc_set.ei_k:
                                print("cancel open_order by ei_k")
                                break

                        time.sleep(config.init_set.realtime_term)  # <-- for realtime price function

                        #       Todo        #
                        #        1. breakout_gty_ratio
                        #           a. exec_open_qty 와 open_quantity 의 ratio 비교
                        #           b. realtime_term 에 delay 주지 않기 위해, term 설정
                        #        2. 이걸 설정 안해주면 체결되도 close_order 진행 안됨
                        if first_exec_qty_check or time.time() - check_time >= config.ep_set.qty_check_term:

                            try:
                                exec_open_quantity = get_remaining_quantity(config.init_set.symbol)
                            except Exception as e:
                                print('error in exec_open_quantity check :', e)
                                time.sleep(config.init_set.api_retry_term)
                                continue
                            else:
                                first_exec_qty_check = 0
                                check_time = time.time()

                            if abs(exec_open_quantity / open_quantity) >= config.ep_set.breakout_qty_ratio:
                                break

                #           1. when, open order time expired or executed              #
                #           2. regardless to position exist, cancel all open orders       #
                while 1:
                    try:
                        request_client.cancel_all_orders(symbol=config.init_set.symbol)
                        print()
                    except Exception as e:
                        print('error in cancel remaining open order :', e)
                    else:
                        break

                if orderside_changed:
                    first_iter = False
                    print('orderside_changed :', orderside_changed)
                    print()
                    continue
                else:
                    break

            #       reject unacceptable amount of asset     3
            if available_balance < self.min_balance:
                continue

            while 1:  # <-- for complete exec_open_quantity function
                try:
                    exec_open_quantity = get_remaining_quantity(config.init_set.symbol)
                except Exception as e:
                    print('error in exec_open_quantity check :', e)
                    continue
                break

            if exec_open_quantity == 0.0:  # <-- open order 가 체결되어야 position 이 생김

                income = 0
                # continue

            #       position / 체결량 이 존재하면 close order 진행      #
            else:

                print('open order executed')

                #        1. save trade_log      #
                #        2. check str(res_df.index[-2]), back_sync entry on close     #
                entry_timeindex = str(res_df.index[-2])
                trade_log[entry_timeindex] = [ep, open_side, "open"]

                with open(trade_log_path + logger_name, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    print("entry trade_log dumped !")

                print()

                # ----------- close order ----------- #

                #           set close side          #
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                load_new_df2 = 1
                limit_tp = 0
                remained_tp_id = None
                prev_remained_tp_id = None
                tp_exec_dict = {}

                while 1:

                    #        1-1. load new_df in every new_df2 index change           #
                    #             -> 이전 index 와 비교하려면, 어차피 load_new_df 해야함      #

                    if not config.tp_set.static_tp or not config.out_set.static_out:
                        
                        if load_new_df2:    # dynamic_out & tp phase
    
                            try:
                                res_df_list = []
                                old_df = 0
                                for itv_i, interval_ in enumerate(config.init_set.interval_list):

                                    if interval_ != "None":
                                        new_df_, _ = concat_candlestick(config.init_set.symbol, interval_,
                                                                        days=1,
                                                                        limit=config.init_set.row_list[itv_i],
                                                                        timesleep=0.2,
                                                                        show_process=False)
                                        if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                            old_df = 1
                                            break
                                        else:
                                            new_df = new_df_.iloc[-config.init_set.row_list[itv_i]:].copy()
                                    else:
                                        new_df = None

                                    res_df_list.append(new_df)

                                if old_df:
                                    continue

                            except Exception as e:
                                print('error in get new_dfs (load_new_df2 phase) :', e)
                                continue

                            else:
                                load_new_df2 = 0

                            try:
                                res_df = self.utils.sync_check(res_df_list)

                            except Exception as e:
                                print("error in sync_check (load_new_df2 phase) :", e)
                                continue

                            try:
                                res_df = self.utils.enlist_tr(res_df, config)
                                print('res_df.index[-1] (load_new_df2 phase) :', res_df.index[-1])
                                # print()

                            except Exception as e:
                                print('error in enlist_tr :', e)
                                continue

                            #        dynamic inversion platform adjusted         #
                            if open_side == OrderSide.BUY:
                                if config.ep_set.short_inversion:
                                    out_series = res_df['short_tp']
                                    tp_series = res_df['short_out']
                                else:
                                    out_series = res_df['long_out']
                                    tp_series = res_df['long_tp']
                            else:
                                if config.ep_set.long_inversion:
                                    out_series = res_df['long_tp']
                                    tp_series = res_df['long_out']
                                else:
                                    out_series = res_df['short_out']
                                    tp_series = res_df['short_tp']

                            if not config.out_set.static_out:
                                out = out_series.iloc[config.init_set.last_index]

                            if not config.tp_set.static_tp:
                                tp = tp_series.iloc[config.init_set.last_index]

                    # --------- limit tp order 여부 조사 phase --------- #
                    #          1. static_limit 은 reorder 할필요 없음
                    if config.tp_set.tp_type == OrderType.LIMIT:

                        prev_remained_tp_id = remained_tp_id

                        try:
                            remained_tp_id = remaining_order_check(config.init_set.symbol)

                        except Exception as e:
                            print('error in remaining_order_check :', e)
                            time.sleep(config.init_set.api_retry_term)
                            continue

                        if config.tp_set.static_tp:
                            if remained_tp_id is None:  # first tp_order
                                limit_tp = 1
                        else:
                            if remained_tp_id is None:  # first tp_order
                                limit_tp = 1
                            else:
                                #       tp 가 달라진 경우만 진행 = reorder     #
                                if tp_series.iloc[config.init_set.last_index] != tp_series.iloc[config.init_set.last_index - 1]:
                                    limit_tp = 1

                                    #           미체결 tp order 모두 cancel               #
                                    #           regardless to position exist, cancel all tp orders       #
                                    while 1:
                                        try:
                                            request_client.cancel_all_orders(symbol=config.init_set.symbol)
                                        except Exception as e:
                                            print('error in cancel remaining tp order :', e)
                                            time.sleep(config.init_set.api_retry_term)
                                            continue
                                        else:
                                            break

                    # ------- limit_tp order phase ------- #
                    if limit_tp:

                        while 1:
                            #         get realtime price, volume precision       #
                            try:
                                price_precision, quantity_precision = get_precision(config.init_set.symbol)
                            except Exception as e:
                                print('error in get price & volume precision :', e)
                                continue
                            else:
                                print('price_precision :', price_precision)
                                print('quantity_precision :', quantity_precision)
                                break

                        #        1. prev_remained_tp_id 생성
                        #       Todo        #
                        #        2. (partial) tp_list 를 직접 만드는 방법 밖에 없을까 --> 일단은.
                        #        3. tp_list 는 ultimate_tp 부터 정렬        #
                        # tp_list = [tp, tp2]
                        # tp_series_list = [tp_series, tp2_series]
                        tp_list = [tp]

                        if len(tp_list) > 1:    # if partial_tp,

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
                                    print("executed_sub_prev_tp, executed_sub_tp :",
                                          executed_sub_prev_tp, executed_sub_tp)
                                    tp_exec_dict[res_df.index[-2]] = [executed_sub_prev_tp, executed_sub_tp]

                                    #        pop sub_tp        #
                                    tp_list.pop()
                                    # tp_series_list.pop()
                                    print("tp_list.pop() executed")

                        tp_list = list(map(lambda x: calc_with_precision(x, price_precision), tp_list))
                        print('tp_list :', tp_list)

                        while 1:
                            try:
                                result = partial_limit(config.init_set.symbol, tp_list, close_side, quantity_precision,
                                                       config.tp_set.partial_qty_divider)

                                if result is not None:
                                    print(result)
                                    time.sleep(config.init_set.api_retry_term)
                                    continue

                            except Exception as e:
                                print('error in partial_limit :', e)
                                time.sleep(config.init_set.api_retry_term)
                                continue

                            else:
                                print("limit tp order enlisted :", datetime.now())
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
                        if config.tp_set.tp_type == OrderType.LIMIT:
                            try:
                                tp_remain_quantity = get_remaining_quantity(config.init_set.symbol)
                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                time.sleep(config.init_set.api_retry_term)
                                continue

                            #       들고 있는 position quantity 가 0 이면, 이 거래를 끝임        #
                            #        1. if tp_remain_quantity < 1 / (10 ** quantity_precision): 로 변경되어야할 것
                            # if tp_remain_quantity == 0.0:
                            if tp_remain_quantity <= 1 / (10 ** quantity_precision):
                                # if tp_remain_quantity < 1 / (10 ** quantity_precision):
                                print('all limit tp order executed')

                                #            1. market tp / out 에 대한 log 방식도 서술
                                #            2. dict 사용 이유 : partial_tp platform
                                if not config.tp_set.static_tp:
                                    tp_exec_dict[res_df.index[-2]] = [tp_series.iloc[config.init_set.last_index - 1], tp]
                                else:
                                    tp_exec_dict[res_df.index[-2]] = [tp]
                                print("tp_exec_dict :", tp_exec_dict)

                                limit_done = 1
                                break

                        # --------- market close signal check phase (out | tp) --------- #
                        try:

                            #        1. 1m df 는 분당 1번만 진행
                            #         a. 그렇지 않을 경우 realtime_price latency 에 영향 줌

                            #        back_pr wait_time 과 every minute end loop 의 new_df_ 를 위해 필요함
                            if config.tp_set.static_tp and config.out_set.static_out:

                                # if config.tp_set.tp_type == 'LIMIT' and config.out_set.hl_out:
                                #     pass
                                # else:
                                try:
                                    if load_new_df3:
                                        new_df_, _ = concat_candlestick(config.init_set.symbol, config.init_set.interval_list[0], days=1, limit=config.init_set.row_list[0],
                                                                        timesleep=0.2)
                                        if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                            continue

                                except Exception as e:
                                    print('error in load_new_df3 :', e)
                                    #        1. adequate term for retries
                                    time.sleep(config.init_set.api_retry_term)
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
                                            print("out :", out)

                                    #       long close_out       #
                                    else:
                                        if new_df_['close'].iloc[config.init_set.last_index] <= out:
                                            market_close_on = True
                                            log_tp = new_df_['close'].iloc[config.init_set.last_index]
                                            print("out :", out)

                                #       long market tp       #
                                if config.tp_set.tp_type == "MARKET":
                                    if new_df_['close'].iloc[config.init_set.last_index] >= tp:
                                        market_close_on = True
                                        log_tp = new_df_['close'].iloc[config.init_set.last_index]
                                        print("tp :", tp)

                            else:
                                if config.out_set.use_out:
                                    #       short hl_out      #
                                    if config.out_set.hl_out:
                                        if realtime_price >= out:
                                            market_close_on = True
                                            log_tp = out
                                            print("out :", out)

                                    #       short close_out     #
                                    else:
                                        if new_df_['close'].iloc[config.init_set.last_index] >= out:
                                            market_close_on = True
                                            log_tp = new_df_['close'].iloc[config.init_set.last_index]
                                            print("out :", out)

                                #       short market tp      #
                                if config.tp_set.tp_type == "MARKET":
                                    if new_df_['close'].iloc[config.init_set.last_index] <= tp:
                                        market_close_on = True
                                        log_tp = new_df_['close'].iloc[config.init_set.last_index]
                                        print("tp :", tp)

                            if market_close_on:
                                if config.out_set.hl_out:
                                    print("realtime_price :", realtime_price)
                                print("market_close_on is True")

                                # --- log tp --- #
                                tp_exec_dict[new_df_.index[-2]] = [log_tp]
                                print("tp_exec_dict :", tp_exec_dict)
                                break

                        except Exception as e:
                            print('error in checking market close signal :', e)
                            continue

                        # ------ minute end phase ------ #
                        if datetime.now().timestamp() > datetime.timestamp(new_df_.index[-1]):

                            if not config.tp_set.static_tp or not config.out_set.static_out:
                                load_new_df2 = 1
                                break
                            else:
                                # if config.tp_set.tp_type == 'LIMIT' and config.out_set.hl_out:
                                #     pass
                                # else:
                                load_new_df3 = 1

                        else:
                            time.sleep(config.init_set.realtime_term)

                    #   all limit close executed     #
                    if limit_done:
                        fee += config.init_set.limit_fee
                        break

                    # if load_new_df2:    # new_df 와 dynamic limit_tp reorder 가 필요한 경우
                    #     continue

                    if not market_close_on:  # = load_new_df2
                        continue

                    #               market close phase - get market signal               #
                    else:   # market_close_on = 1

                        fee += config.init_set.market_fee

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
                r_qty = 1   # base asset ratio
                # ---------- calc in for loop ---------- #
                for q_i, (k_ts, v_tp) in enumerate(sorted(tp_exec_dict.items(), key=lambda x: x[0], reverse=True)):

                    # ---------- get real_tp phase ---------- #
                    if len(v_tp) > 1:
                        prev_tp, tp = v_tp

                        if prev_tp == tp:
                            real_tp = tp

                        else:   # dynamic_tp 로 인한 open tp case
                            if close_side == OrderSide.BUY:
                                if tp > res_df['open'].loc[k_ts]:
                                    real_tp = res_df['open'].loc[k_ts]
                                    print("market tp executed !")
                                else:
                                    real_tp = tp
                            else:
                                if tp < res_df['open'].loc[k_ts]:
                                    real_tp = res_df['open'].loc[k_ts]
                                    print("market tp executed !")
                                else:
                                    real_tp = tp

                    else:   # market close's real_tp | out (partial_tp also have market out)
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

                    print("real_tp :", real_tp)
                    print("ep :", ep)
                    print("fee :", fee)
                    print("temp_qty :", temp_qty)

                    #            save exit data         #
                    #            exit_timeindex use "-1" index           #
                    # exit_timeindex = str(res_df.index[-1])
                    # trade_log[exit_timeindex] = [real_tp, "close"]
                    trade_log[k_ts] = [real_tp, "close"]

                with open(trade_log_path + logger_name, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    print("exit trade_log dumped !")

                end_timestamp = int(time.time() * 1000)

                #           get total income from this trade       #
                while 1:
                    try:
                        income = total_income(config.init_set.symbol, start_timestamp, end_timestamp)
                        self.accumulated_income += income
                        print("income :", income, "USDT")

                    except Exception as e:
                        print('error in total_income :', e)

                    else:
                        break

                tmp_profit = income / real_balance * leverage
                self.accumulated_profit *= (1 + tmp_profit)
                self.calc_accumulated_profit *= 1 + (calc_tmp_profit - 1) * leverage
                print('temporary profit : %.3f (%.3f) %%' % (tmp_profit * 100, (calc_tmp_profit - 1) * leverage * 100))
                print('accumulated profit : %.3f (%.3f) %%' % (
                    (self.accumulated_profit - 1) * 100, (self.calc_accumulated_profit - 1) * 100))
                print('accumulated income :', self.accumulated_income, "USDT")
                print()
