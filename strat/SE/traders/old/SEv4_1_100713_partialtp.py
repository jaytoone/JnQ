import os

# switch_path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend"

dir_path = os.path.abspath('../')
print("dir_path :", dir_path)

# os.chdir(switch_path)
os.chdir("../../")
# print("os.getcwd() :", os.getcwd())
# quit()

from binance_futures_modules import *
# from funcs.funcs_for_trade import *
from binance_futures_concat_candlestick import concat_candlestick
from easydict import EasyDict
from SE.utils.v4_1 import *
import pickle

#       Todo        #
#        1. check upper dir


class Trader:

    def __init__(self, symbol, interval, interval2, interval3, leverage_, initial_asset, save_stacked_df=False, stacked_df_exist=True):

        #       basic param     #
        self.last_index = -2
        self.symbol = symbol
        self.interval = interval
        self.interval2 = interval2
        self.interval3 = interval3
        self.use_rows = 200  # <-- important var. for "Trader" latency
        self.use_rows2 = 100
        self.use_rows3 = 100  # Todo : 이걸로 되는지 모르겠음
        # self.threshold = threshold
        # self.tp_gap = tp_gap
        self.leverage = leverage_
        self.initial_asset = initial_asset
        self.save_stacked_df = save_stacked_df
        self.stacked_df_exist = stacked_df_exist
        self.over_balance = None
        self.min_balance = 5.0  # USDT

        # ------- ep set ------- #
        self.static_ep = 1
        self.ep_gap = 0.5  # st_gap is critera # 지정한 line (middle) 부터 st_gap 적용
        self.ep_protect_gap = 0.2
        # ep_out_gap = 0.5 # ep ~ out 사이의 gap, 이 안에 들 경우 진입하도록 (max_tr 을 위해 적용함) --> 필요없어질 것
        self.entry_incycle = 0

        # ------- out set ------- #
        self.use_out = 1
        self.static_out = 1

        self.hl_out = 0

        self.price_restoration = 0

        self.retouch = 0  # out_line 에 대한 retouch 를 out 으로 한건지에 대한 여부
        self.retouch_out_period = 500
        self.second_out = 0  # 기존 out 과는 다른 second_out 사용 여부

        self.out_gap = 0.25
        self.approval_st_gap = 1.5
        self.second_out_gap = 0.5  # static_out 을 위한 gap

        # ------- tp set ------- #
        self.pgfr = 0.3
        self.static_tp = 0

        self.tp_gap =2.5  # st_gap is critera

        self.partial_num = 1
        self.partial_qty_divider = 1.5

        # ------- lvrg set ------- #
        self.static_lvrg = 1
        self.target_pct = 0.05

        #       pr param        #
        self.trading_fee = 0.0008
        self.accumulated_income = 0.0
        self.calc_accumulated_profit = 1.0
        self.accumulated_profit = 1.0
        self.accumulated_back_profit = 1.0

        sub_client.subscribe_aggregate_trade_event(symbol.lower(), callback, error)
        self.sub_client = sub_client

    def run(self):

        df_path = dir_path + '/%s.xlsx' % self.symbol

        #         1. leverage type => "isolated"          #
        try:
            request_client.change_margin_type(symbol=self.symbol, marginType=FuturesMarginType.ISOLATED)

        except Exception as e:

            print('error in change_margin_type :', e)
        else:
            print('leverage type --> isolated')

        #         2. confirm limit leverage          #
        try:
            limit_leverage = get_limit_leverage(symbol_=self.symbol)

        except Exception as e:
            print('error in get_limit_leverage :', e)
            quit()

        else:
            print('limit leverage :', limit_leverage)

        print()
        print('# ----------- Trader v1 ----------- #')

        #        1. save open / close data to dict     #
        #        2. dict name = str(datetime.now().timestamp()).split(".")[0]      #
        #        3. key = str(timeindex), value = [ep, ordertype] | [tp (=exit_price)]
        #        4. logger_name is the trade start time index        #
        # logger_name = "%s.pkl" % str(datetime.now()).split(".")[0]  # <-- invalid arguments
        logger_name = "%s.pkl" % str(datetime.now().timestamp()).split(".")[0]
        trade_log = {}

        while 1:

            #       load config       #
            with open(dir_path + '/config.json', 'r') as cfg:
                config = EasyDict(json.load(cfg))

            fundamental = config.FUNDMTL
            order = config.ORDER

            #       check run        #
            if not fundamental.run:
                continue

            #       init order side      #
            open_side = None
            # close_side = None

            #       get startTime     #
            start_timestamp = int(time.time() * 1000)
            entry_const_check = True
            # load_new_df = False   # entry_const_check 과 겹침

            while 1:

                if entry_const_check:

                    #       log last trading time     #
                    #       미체결을 고려해, entry_const_check 마다 log 수행       #
                    trade_log["last_trading_time"] = str(datetime.now())
                    with open(dir_path + "/trade_log/" + logger_name, "wb") as dict_f:
                        pickle.dump(trade_log, dict_f)

                    temp_time = time.time()

                    # if self.stacked_df_exist:

                    try:

                        new_df_, _ = concat_candlestick(self.symbol, self.interval, days=1, limit=self.use_rows,
                                                        timesleep=0.2, show_process=False)
                        #       Todo        #
                        #        1. 모든 inveral 의 마지막 time_index 은 현재시간보다 커야함 (최신이라면)        #
                        if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                            # load_new_df = True
                            continue

                        new_df2_, _ = concat_candlestick(self.symbol, self.interval2, days=1, limit=self.use_rows2,
                                                         timesleep=0.2)
                        if datetime.timestamp(new_df2_.index[-1]) < datetime.now().timestamp():
                            continue

                        new_df3_, _ = concat_candlestick(self.symbol, self.interval3, days=1, limit=self.use_rows3,
                                                         timesleep=0.2)
                        if datetime.timestamp(new_df3_.index[-1]) < datetime.now().timestamp():
                            continue

                        new_df = new_df_.iloc[-self.use_rows:].copy()
                        new_df2 = new_df2_.iloc[-self.use_rows2:].copy()
                        new_df3 = new_df3_.iloc[-self.use_rows3:].copy()
                        print("complete load_df execution time :", datetime.now())

                    except Exception as e:
                        print("error in load new_dfs :", e)
                        continue

                    # if self.save_stacked_df:
                    #
                    #     if self.stacked_df_exist:
                    #
                    #         try:
                    #             stacked_df = pd.read_excel(df_path, index_col=0)
                    #
                    #             #    use rows   #
                    #             #       add ep columns for append      #
                    #             # new_df['ep'] = np.nan
                    #             prev_complete_df = stacked_df.iloc[:-1, :]
                    #             stacked_df = prev_complete_df.append(new_df)
                    #
                    #             stacked_df = stacked_df[~stacked_df.index.duplicated(keep='first')].iloc[
                    #                          -order.use_rows:, :]
                    #
                    #         except Exception as e:
                    #             print("error in making stacked_df :", e)
                    #             continue
                    #
                    #     else:
                    #
                    #         try:
                    #             # if not self.stacked_df_exist:
                    #             days = calc_train_days(interval=self.interval, use_rows=order.use_rows)
                    #             stacked_df, _ = concat_candlestick(self.symbol, self.interval, days=days, timesleep=0.2)
                    #             stacked_df = stacked_df.iloc[-order.use_rows:, :]
                    #             self.stacked_df_exist = True
                    #
                    #             stacked_df.to_excel(df_path)
                    #
                    #         except Exception as e:
                    #             print("error in load new stacked_df :", e)
                    #             self.stacked_df_exist = False
                    #             continue

                    print("~ load_new_df time : %.2f" % (time.time() - temp_time))
                    # temp_time = time.time()

                    try:
                        #       latency 를 줄이기 위해서는, input df length 를 조절해야함     #
                        res_df = sync_check(new_df, new_df2, new_df3)

                        print('~ sync_check time : %.5f' % (time.time() - temp_time))

                    except Exception as e:
                        print("error in sync_check :", e)
                        continue

                    #        Todo       #
                    #         ------------ ep / out / tp ------------ #
                    #           1. 추후 limit_entry 경우, ep 추가 고려
                    #           2. 현재, 완성된 close 기준 data 사용 진행중
                    try:
                        entry, short_ep, long_ep, \
                        short_out, long_out, short_tp, long_tp = enlist_epouttp(res_df, self.tp_gap)
                        print('res_df.index[-1] :', res_df.index[-1])
                        print('~ ep enlist time : %.5f' % (time.time() - temp_time))
                        print()

                    except Exception as e:
                        print('error in enlist_epouttp :', e)
                        entry_const_check = True  # res_df 에 data 가 덮여씌워질 것임
                        continue

                    # ------------ open order constraints phase ------------ #
                    try:

                        #        save res_df before, entry const check      #
                        if order.df_log:
                            excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                            res_df.to_excel(dir_path + "/df_log/%s.xlsx" % excel_name)

                        # ---------------------- short ---------------------- #
                            # ----------- major_markup ----------- #
                                # ----------- cbline ----------- #
                        #       Todo        #
                        #        1. cbline const. compare with close, so 'close' data should be confirmed
                        #        2. or, entry market_price would be replaced with 'close'
                        if res_df['close'].iloc[self.last_index] <= res_df['cloud_bline_3m'].iloc[self.last_index]:

                            # ----------- minor_markup ----------- #
                            if entry[self.last_index] == -1:

                                # -------------- ep scheduling -------------- #
                                    # ----- st_gap ----- #
                                if res_df['close'].iloc[self.last_index] \
                                    >= res_df['st_lower2_5m'].iloc[self.last_index] + self.ep_protect_gap * res_df['st_gap_5m'].iloc[self.last_index]:

                                # -------------- tp scheduling -------------- #
                                    # ----- pgfr ----- #
                                    #       Todo        #
                                    #        1. short_tp / out 중에 결정하는게 의미가 있을까
                                    #         a. pgfr 의의는 tp_gap & fee ratio, short_tp 사용하는게 맞을 것
                                    if (short_tp.iloc[self.last_index] / res_df['close'].iloc[self.last_index] - self.trading_fee) / abs(
                                            short_tp.iloc[self.last_index] / res_df['close'].iloc[self.last_index] + self.trading_fee) > self.pgfr:

                                        open_side = OrderSide.BUY   # inversion

                        # ---------------------- long ---------------------- #
                            # ----------- major_markup ----------- #
                                # ----------- cbline ----------- #
                        if res_df['close'].iloc[self.last_index] >= res_df['cloud_bline_3m'].iloc[self.last_index]:

                            # ----------- minor_markup ----------- #
                            if entry[self.last_index] == 1:

                                # -------------- ep scheduling -------------- #
                                    # ----- st_gap ----- #
                                if res_df['close'].iloc[self.last_index] \
                                        <= res_df['st_upper2_5m'].iloc[self.last_index] - self.ep_protect_gap * \
                                        res_df['st_gap_5m'].iloc[self.last_index]:

                                # -------------- tp scheduling -------------- #
                                    # ----- pgfr ----- #
                                    if (long_tp.iloc[self.last_index] / res_df['close'].iloc[self.last_index] - self.trading_fee) / abs(
                                            long_tp.iloc[self.last_index] / res_df['close'].iloc[self.last_index] + self.trading_fee) > self.pgfr:

                                        open_side = OrderSide.SELL

                    except Exception as e:
                        print("error in open const phase :", e)
                        continue

                    entry_const_check = False   # once used, turn off const_check

                check_entry_sec = datetime.now().second
                if open_side is not None:

                    #       Todo        #
                    #        1. const. open market second
                    if check_entry_sec > 3:
                        open_side = None
                        print("check_entry_sec :", check_entry_sec)
                        print()
                        continue    # entry_const_check = False 라서 open_side None phase 로 감

                    break

                    #   1. 추후 position change platform 으로 변경 가능
                    #    a. first_iter 사용

                # ------- open_side is None [ waiting zone ] ------- #
                else:
                    time.sleep(fundamental.realtime_term)  # <-- term for realtime market function

                    #       check tick change      #
                    #       latest df confirmation       #
                    if datetime.timestamp(res_df.index[-1]) < datetime.now().timestamp():
                        entry_const_check = True
                        print('res_df[-1] timestamp :', datetime.timestamp(res_df.index[-1]))
                        print('current timestamp :', datetime.now().timestamp())
                        print()

                    # time.sleep(fundamental.close_complete_term)   # <-- term for close completion

            # print('realtime_price :', realtime_price)

            first_iter = True  # 포지션 변경하는 경우, 필요함
            while 1:  # <-- loop for 'check order type change condition'

                print('open_side :', open_side)

                # if order.entry_type == OrderType.MARKET:
                #     #             ep with market            #
                #     try:
                #         ep = get_market_price_v2(self.sub_client)
                #     except Exception as e:
                #         print('error in get_market_price :', e)
                #         continue
                # else:

                #             ep limit & market open             #
                if open_side == OrderSide.BUY:
                    ep = short_ep.iloc[self.last_index]  # inversion mode adjusted
                    out = short_out.iloc[self.last_index]
                    tp = short_tp.iloc[self.last_index]
                else:
                    ep = long_ep.iloc[self.last_index]
                    out = long_out.iloc[self.last_index]
                    tp = long_tp.iloc[self.last_index]

                leverage = self.leverage

                #         get price, volume precision --> reatlime 가격 변동으로 인한 precision 변동 가능성       #
                try:
                    price_precision, quantity_precision = get_precision(self.symbol)
                except Exception as e:
                    print('error in get price & volume precision :', e)
                    continue
                else:
                    print('price_precision :', price_precision)
                    print('quantity_precision :', quantity_precision)

                ep = calc_with_precision(ep, price_precision)
                out = calc_with_precision(out, price_precision)  # this means static_out
                tp = calc_with_precision(tp, price_precision)   # this means static_tp
                leverage = min(limit_leverage, leverage)
                print('ep :', ep)
                print('out :', out)
                print('tp :', tp)
                print('leverage :', leverage)
                print('~ ep out lvrg set time : %.5f' % (time.time() - temp_time))

                try:
                    request_client.change_initial_leverage(symbol=self.symbol, leverage=leverage)
                except Exception as e:
                    print('error in change_initial_leverage :', e)
                    continue  # -->  ep market 인 경우에 조심해야함
                else:
                    print('leverage changed -->', leverage)

                if first_iter:
                    #          define start asset          #
                    if self.accumulated_income == 0.0:

                        available_balance = self.initial_asset  # USDT

                    else:
                        available_balance += income

                #          get availableBalance          #
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
                    continue

                print('~ get balance time : %.5f' % (time.time() - temp_time))

                #          get available open_quantity         #
                open_quantity = available_balance / ep * leverage
                open_quantity = calc_with_precision(open_quantity, quantity_precision)
                if self.over_balance is not None:
                    print('available_balance (temp) :', available_balance)
                else:
                    print('available_balance :', available_balance)

                print('open_quantity :', open_quantity)

                #           open order            #
                orderside_changed = False
                open_retry_cnt = 0


                code_1111 = 0
                while 1:  # <-- loop for complete open order

                    try:
                        if order.entry_type == OrderType.MARKET:

                                #           market order            #
                                request_client.post_order(symbol=self.symbol, side=open_side, ordertype=OrderType.MARKET,
                                                          quantity=str(open_quantity))

                        else:
                                #       limit order            #
                                #       limit order needs execution waiting time & check remaining order         #

                                request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                                                   side=open_side,
                                                                   ordertype=order.entry_type,
                                                                   quantity=str(open_quantity), price=str(ep),
                                                                   reduceOnly=False)
                    except Exception as e:
                        print('error in limit open order :', e)

                        #        1. price precision validation      #
                        if '-4014' in str(e):
                            try:
                                realtime_price = get_market_price_v2(self.sub_client)
                                price_precision = calc_precision(realtime_price)
                                ep = calc_with_precision(ep, price_precision)
                                print('modified price & precision :', ep, price_precision)

                            except Exception as e:
                                print('error in get_market_price :', e)

                            continue

                        #        -1111 : Precision is over the maximum defined for this asset   #
                        #         = quantity precision error        #
                        if '-1111' in str(e):
                            code_1111 += 1
                            try:
                                _, quantity_precision = get_precision(self.symbol)
                                if code_1111 % 3 == 0:
                                    quantity_precision -= 1
                                elif code_1111 % 3 == 2:
                                    quantity_precision += 1

                                quantity = available_balance / ep * leverage
                                quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')
                                print('modified qty & precision :', quantity, quantity_precision)

                            except Exception as e:
                                print('error in get modified qty_precision :', e)

                            continue

                        open_retry_cnt += 1

                        #        -2019 : Margin is insufficient     #
                        if '-2019' in str(e):
                            try:
                                max_available_balance = get_availableBalance()
                                #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                                self.over_balance = available_balance
                                available_balance = max_available_balance * 0.9
                                #          Get available quantity         #
                                quantity = available_balance / ep * leverage
                                quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')
                                print('available_balance (temp) :', available_balance)
                                print('quantity :', quantity)

                            except Exception as e:
                                print('error in get_availableBalance() :', e)

                        elif open_retry_cnt > 100:
                            print('open_retry_cnt over 100')
                            quit()
                        continue

                    else:
                        print('open order enlisted :', datetime.now())
                        break

                # -------- execution wait time -------- #
                if order.entry_type == OrderType.MARKET:
                    #       enough time for open_quantity consumed      #
                    time.sleep(60 - fundamental.bar_close_second)
                else:
                    #       limit only, order execution wait time & check breakout_qty_ratio         #
                    while 1:

                        #       temporary : 해당 종가까지 체결 대기       #
                        #       datetime.timestamp(res_df.index[-1]) -> 2분이면, 01:59:999 이런식으로 될것
                        if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                            # if datetime.now().timestamp() + order.entry_execution_wait > datetime.timestamp(res_df.index[-1]):
                            break

                        time.sleep(fundamental.realtime_term)  # <-- for realtime price function

                #           1. when, open order time expired or executed              #
                #           2. regardless to position exist, cancel all open orders       #
                while 1:
                    try:
                        request_client.cancel_all_orders(symbol=self.symbol)
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

            #       Todo        #
            #        " limit order 가 reduceOnly 인지 잘 확인해야함 "       #
            #        일단은, time_gap 을 두어, executed quantity 검사의 정확도를 높임      #
            #        예외 발생시, 2차 검사 phase 생성     #
            #        ==> 혹시, 여기도 time index 에 종속적인 걸까       #
            time.sleep(1)

            while 1:  # <-- for complete exec_open_quantity function
                try:
                    exec_open_quantity = get_remaining_quantity(self.symbol)
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

                with open(dir_path + "/trade_log/" + logger_name, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    print("entry trade_log dumped !")

                print()

                #           set close side          #
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                #           market : prevent close at open bar           #
                #           Todo            #
                #            1. 더 정확히하려면, time_index confirmation 진행
                if order.entry_type == OrderType.MARKET:
                    time.sleep(60 - datetime.now().second)

                if order.static_tp and order.static_out:
                    load_new_df2 = 0    # dynamic out / tp 인 경우만 진행
                else:
                    load_new_df2 = 1

                #       Todo        #
                #        1. tp 이런식으로 놓는거 좋지 않음      #
                remained_tp_id = None
                prev_remained_tp_id = None
                tp_exec_dict = {}

                #   static 인 경우, open 에서 제시한 out / tp data 로 거래 진행함
                #   아래 phase 는 static & dynamic limit 모두 호환해야함
                while 1:

                    #        0. websocket 으로 load_new_df 가능할지 확인             #
                    #        1-1. load new_df in every new_df2 index change           #
                    #             -> 이전 index 와 비교하려면, 어차피 load_new_df 해야함      #
                    #        2. reorder tp 경우, enlist 된 기존 tp cancel 해야함
                    #        2-1. new_df check 동일하게 수행함

                    #       Todo        #
                    #        1. dynamic_tp 의 경우만 load_new_df2 필요함
                    #         a. dynamic_limit & dynamic_market tp
                    #        2. static_limit & market tp 는 ?
                    #         a. limit 은 limit order 만 진행
                    #         b. market 은 market order 만 진행
                    #        3. dynamic_out 도 있음
                    if not order.static_tp or not order.static_out:
                        
                        if load_new_df2:    # dynamic_out & tp phase
    
                            try:
                                new_df_, _ = concat_candlestick(self.symbol, self.interval, days=1, limit=self.use_rows,
                                                                timesleep=0.2)
                                if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                    continue
    
                                new_df2_, _ = concat_candlestick(self.symbol, self.interval2, days=1, limit=self.use_rows2,
                                                                 timesleep=0.2)
                                if datetime.timestamp(new_df2_.index[-1]) < datetime.now().timestamp():
                                    continue
    
                                new_df3_, _ = concat_candlestick(self.symbol, self.interval3, days=1, limit=self.use_rows3,
                                                                 timesleep=0.2)
                                if datetime.timestamp(new_df3_.index[-1]) < datetime.now().timestamp():
                                    continue
    
                                new_df = new_df_.iloc[-self.use_rows:].copy()
                                new_df2 = new_df2_.iloc[-self.use_rows2:].copy()
                                new_df3 = new_df3_.iloc[-self.use_rows3:].copy()
    
                                res_df = sync_check(new_df, new_df2, new_df3)

                                try:
                                    _, _, _, \
                                    short_out, long_out, short_tp, long_tp = enlist_epouttp(res_df, self.tp_gap)
                                    print('res_df.index[-1] (load_new_df2 phase) :', res_df.index[-1])
                                    # print()

                                except Exception as e:
                                    print('error in enlist_epouttp :', e)
                                    continue

                                if open_side == OrderSide.BUY:
                                    out_series = short_out   # inversion mode adjusted
                                    tp_series = short_tp
                                else:
                                    out_series = long_out
                                    tp_series = long_tp

                                out = out_series.iloc[self.last_index]
                                tp = tp_series.iloc[self.last_index]

                            except Exception as e:
                                print('error in get new_dfs (load_new_df2 phase) :', e)
                                continue
    
                            else:
                                load_new_df2 = False
    
                        # ----------- for dynamic_tp phase only - check tp change ----------- #
                        #         0. we don't need this phase for static_tp
                        #         1. tp_changed, survey remaining tp orders
                        #          a. if exists, cancel them all (market & limit)
                        if not order.static_tp:

                            if tp_series.iloc[self.last_index] != tp_series.iloc[self.last_index - 1]:

                                #       Todo        #
                                #        1. check, if limit_order exist

                                #       cancel order 전에, sub_tp 체결여부를 확인            #
                                #       tp_id 를 이곳에서 survey 하는 이유는, id 를 통해 cancel 여부를 결정하기 때문임
                                #       갱신되기전 tp_id list 를 prev_tp_id list 에 저장
                                prev_remained_tp_id = remained_tp_id

                                try:
                                    remained_tp_id = remaining_order_check(self.symbol)

                                except Exception as e:
                                    print('error in remaining_order_check :', e)
                                    continue

                                if remained_tp_id is not None:

                                    #           미체결 tp order 모두 cancel               #
                                    #           regardless to position exist, cancel all tp orders       #
                                    try:
                                        request_client.cancel_all_orders(symbol=self.symbol)
                                    except Exception as e:
                                        print('error in cancel remaining tp order :', e)
                                        continue

                    #         Todo          #
                    #          limit tp order enlist phase (dynamic & static)
                    if order.tp_type == OrderType.LIMIT:

                        #         get price, volume precision -> reatlime 가격 변동으로 인한 precision 변동 가능성       #
                        try:
                            price_precision, quantity_precision = get_precision(self.symbol)
                        except Exception as e:
                            print('error in get price & volume precision :', e)
                            continue
                        else:
                            print('price_precision :', price_precision)
                            print('quantity_precision :', quantity_precision)

                        #            tp_list 는 ultimate_tp 부터 정렬     #
                        #            1. 체결된 tp 에 대해서는 tp_list 에서 제외       #
                        #            2. 체결되지 않은 tp 에 대해서 static 한 경우 취소할 필요가 없을건데,
                        #               static 여부는 tp_list 내부 모든 tp 에 동일하게 적용될 거니까 일단 무시
                        #            3. 체결 검수 부분만 한번더 확인
                        #             4. static 이면 어뜩함 ?           #
                        #             4-1. 어차피 reorder 할 필요가 없으면, tp_list 를 수정할 필요가 없음      #

                        #        1. prev_remained_tp_id 생성
                        #       Todo        #
                        #        2. (partial) tp_list 를 직접 만드는 방법 밖에 없을까 --> 일단은.
                        # tp_list = [tp, tp2]
                        # tp_series_list = [tp_series, tp2_series]
                        tp_list = [tp]

                        if len(tp_list) > 1:    # if partial_tp,

                            #       Todo        #
                            #        0. sub_tp 체결시마다 기록
                            #         a. 체결 confirm 은 len(remained_tp_id) 를 통해 진행
                            #        1. back_pr calculation
                            #        위해 체결된 시간 (res_df.index[-2]) 과 해당 tp 를 한번만 기록함        #

                            #       1. pop executed sub_tps from tp_list
                            #       2. log executed sub_tps
                            if remained_tp_id is not None and prev_remained_tp_id is not None:

                                # --- sub_tp execution --- #
                                if remained_tp_id != prev_remained_tp_id:

                                    # --- log sub_tp --- #
                                    executed_sub_tp = tp_series_list[-1].iloc[-2]
                                    executed_sub_prev_tp = tp_series_list[-1].iloc[-3]
                                    print("executed_sub_prev_tp, executed_sub_tp :",
                                          executed_sub_prev_tp, executed_sub_tp)
                                    tp_exec_dict[res_df.index[-2]] = [executed_sub_prev_tp, executed_sub_tp]

                                    # --- pop sub_tp --- #
                                    tp_list.pop()
                                    # tp_series_list.pop()
                                    print("tp_list.pop() executed")

                        tp_list = list(map(lambda x: calc_with_precision(x, price_precision), tp_list))
                        print('tp_list :', tp_list)

                        try:
                            result = partial_limit(self.symbol, tp_list, close_side, quantity_precision,
                                                   self.partial_qty_divider)

                            if result is not None:
                                print(result)
                                continue

                        except Exception as e:
                            print('error in partial_limit :', e)
                            time.sleep(1)
                            continue

                        else:
                            print("limit tp order enlisted :", datetime.now())

                    #            1. limit close (tp) execution check, every minute                 #
                    #            2. check market close signal, simultaneously
                    market_close_on = False
                    log_tp = np.nan
                    load_new_df3 = True
                    while 1:

                        # --------- limit tp execution check phase --------- #
                        if order.tp_type == OrderType.LIMIT:
                            try:
                                tp_remain_quantity = get_remaining_quantity(self.symbol)
                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                continue

                            #       들고 있는 position quantity 가 0 이면, 이 거래를 끝임        #
                            #        1. if tp_remain_quantity < 1 / (10 ** quantity_precision): 로 변경되어야할 것
                            # if tp_remain_quantity == 0.0:
                            if tp_remain_quantity <= 1 / (10 ** quantity_precision):
                                # if tp_remain_quantity < 1 / (10 ** quantity_precision):
                                print('all limit tp order executed')

                                #           Todo            #
                                #            1. market tp / out 에 대한 log 방식도 서술
                                #            2. dict 사용 이유 : partial_tp platform
                                if not order.static_tp:
                                    tp_exec_dict[res_df.index[-2]] = [tp_series.iloc[self.last_index - 1], tp]
                                else:
                                    tp_exec_dict[res_df.index[-2]] = [tp]
                                print("tp_exec_dict :", tp_exec_dict)
                                break

                            if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                                load_new_df2 = True
                                break

                        # --------- market close signal check phase (out | tp) --------- #
                        #          1. 이부분에서 갱신된 res_df 1m data 필요함
                        #           a. confirmed_close 와 out / tp line 을 비교함
                        #          2. market_tp 조건은 tp_type 에 대한 조건을 명시해야함
                        #           a. out 은 market base 로 정의해서 조건문 필요없음
                        try:

                            #       Todo        #
                            #        1. 1m df 는 분당 1번만 진행
                            #         a. 그렇지 않을 경우 realtime_price latency 에 영향 줌
                            if load_new_df3:
                                new_df_, _ = concat_candlestick(self.symbol, self.interval, days=1, limit=self.use_rows,
                                                                timesleep=0.2)
                                if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                    continue

                                load_new_df3 = False

                            else:   # initial load_new_df3 = True
                                #       new_df_ update condition       #
                                if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                    load_new_df3 = True
                                    continue

                            realtime_price = get_market_price_v2(self.sub_client)

                            # ---------- out & tp check ---------- #
                            if open_side == OrderSide.BUY:
                                #       long out (inversion of short)     #
                                if realtime_price <= out:  # hl_out
                                    market_close_on = True
                                    log_tp = out
                                    print("out :", out)

                                #       long tp (inversion of short)     #
                                #       if order.tp_type == OrderType.MARKET,
                                if new_df_['close'].iloc[self.last_index] >= tp:
                                    market_close_on = True
                                    log_tp = tp
                                    print("tp :", tp)

                            else:
                                #       short out (inversion of long)     #
                                if realtime_price >= out:
                                    market_close_on = True
                                    log_tp = out
                                    print("out :", out)

                                #       short tp (inversion of long)     #
                                if new_df_['close'].iloc[self.last_index] <= tp:
                                    market_close_on = True
                                    log_tp = tp
                                    print("tp :", tp)

                            if market_close_on:
                                print("realtime_price :", realtime_price)
                                print("market_close_on is True")
                                # --- log tp --- #
                                tp_exec_dict[new_df_.index[-2]] = [log_tp]
                                print("tp_exec_dict :", tp_exec_dict)
                                break

                        except Exception as e:
                            print('error in checking market close signal :', e)
                            continue

                    if load_new_df2:
                        continue

                    #   all limit close executed     #
                    if not market_close_on:
                        break

                    #               market close phase - get market signal               #
                    else:

                        remain_tp_canceled = False  # 미체결 limit tp 존재가능함
                        while 1:  # <--- This loop for out close & complete close

                            #               cancel all tp order                 #
                            if not remain_tp_canceled:
                                #               remaining tp order check            #
                                try:
                                    remained_orderId = remaining_order_check(self.symbol)

                                except Exception as e:
                                    print('error in remaining_order_check :', e)
                                    continue

                                if remained_orderId is not None:
                                    #           if remained tp order exist, cancel it       #
                                    try:
                                        request_client.cancel_all_orders(symbol=self.symbol)

                                    except Exception as e:
                                        print('error in cancel remaining tp order (out phase) :', e)
                                        continue

                                remain_tp_canceled = True

                            #          get remaining close_remain_quantity         #
                            try:
                                close_remain_quantity = get_remaining_quantity(self.symbol)

                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                continue

                            #           get price, volume precision             #
                            #           -> reatlime 가격 변동으로 인한 precision 변동 가능성       #
                            try:
                                _, quantity_precision = get_precision(self.symbol)

                            except Exception as e:
                                print('error in get price & volume precision :', e)

                            else:
                                print('quantity_precision :', quantity_precision)

                            close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision)
                            print('close_remain_quantity :', close_remain_quantity)

                            #           close remaining close_remain_quantity             #
                            #           1. order side should be opposite side to the open          #
                            #           2. reduceOnly = 'true'       #
                            code_1111 = 0
                            while 1:  # <-- loop for complete close order
                                try:
                                    #       Todo        #
                                    #        1. limit out 은 한동안 없을 것
                                    if order.out_type != OrderType.MARKET:

                                        #           stop limit order            #
                                        # if close_side == OrderSide.SELL:
                                        #     stop_price = calc_with_precision(realtime_price + 5 * 10 ** -price_precision,
                                        #                                      price_precision)
                                        # else:
                                        #     stop_price = calc_with_precision(realtime_price - 5 * 10 ** -price_precision,
                                        #                                      price_precision)

                                        # request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                        #                           side=close_side,
                                        #                           ordertype=order.out_type, stopPrice=str(stop_price),
                                        #                           quantity=str(close_remain_quantity), price=str(realtime_price),
                                        #                           reduceOnly=True)

                                        if pd.isna(out):
                                            exit_price = realtime_price
                                        else:
                                            exit_price = log_tp

                                        #           limit order             #
                                        request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                                                  side=close_side,
                                                                  ordertype=order.out_type,
                                                                  quantity=str(close_remain_quantity), price=str(exit_price),
                                                                  reduceOnly=True)

                                    else:
                                        #           market order            #
                                        request_client.post_order(symbol=self.symbol, side=close_side,
                                                                           ordertype=OrderType.MARKET,
                                                                           quantity=str(close_remain_quantity),
                                                                           reduceOnly=True)

                                except Exception as e:
                                    print('error in close order :', e)

                                    #       check error codes     #
                                    #       Todo        #
                                    #        1. quantity less than zero ?
                                    if '-4003' in str(e):
                                        print('error code check (-4003) :', str(e))
                                        break

                                    #        -1111 : Precision is over the maximum defined for this asset   #
                                    #         = quantity precision error        #
                                    if '-1111' in str(e):
                                        code_1111 += 1
                                        try:
                                            _, quantity_precision = get_precision(self.symbol)
                                            if code_1111 % 3 == 0:
                                                quantity_precision -= 1
                                            elif code_1111 % 3 == 2:
                                                quantity_precision += 1

                                            close_remain_quantity = get_remaining_quantity(self.symbol)
                                            close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision,
                                                                           def_type='floor')
                                            print('modified qty & precision :', close_remain_quantity, quantity_precision)

                                        except Exception as e:
                                            print('error in get modified qty_precision :', e)

                                        continue

                                    order.out_type = OrderType.MARKET   # just 재시도
                                    continue

                                else:
                                    print('market close order enlisted')
                                    break

                            #       enough time for close_remain_quantity to be consumed      #
                            if order.out_type != OrderType.MARKET:

                                #       wait for bar ends      #
                                time.sleep(order.exit_execution_wait - datetime.now().second)
                                print("order.exit_execution_wait - datetime.now().second :",
                                      order.exit_execution_wait - datetime.now().second)
                                print("datetime.now().second :", datetime.now().second)

                            else:
                                time.sleep(1)   # time for qty consumed

                            #               check remaining close_remain_quantity             #
                            try:
                                close_remain_quantity = get_remaining_quantity(self.symbol)
                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                continue

                            if close_remain_quantity == 0.0:
                                print('market close order executed')
                                break

                            else:
                                #           complete close by market            #
                                print('out_type changed to market')
                                order.out_type = OrderType.MARKET
                                continue

                        break  # <--- break for all close order loop, break partial tp loop

                # ----------------------- check back_pr ----------------------- #
                #       total_income() function confirming -> wait close confirm       #
                while 1:

                    if order.tp_type == OrderType.LIMIT:
                        lastest_close_timeindex = res_df.index[-1]
                    else:
                        lastest_close_timeindex = new_df_.index[-1]

                    if datetime.now().timestamp() > datetime.timestamp(lastest_close_timeindex):
                        # try:
                        #     back_df, _ = concat_candlestick(self.symbol, self.interval, days=1, timesleep=0.2)
                        #
                        #     #       realtime candlestick confirmation       #
                        #     if datetime.timestamp(back_df.index[-1]) > datetime.now().timestamp():
                        #         break
                        #
                        # except Exception as e:
                        #     print('error in back-test profit :', e)

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
                            temp_qty = r_qty / self.partial_qty_divider
                        else:
                            temp_qty = r_qty

                    r_qty -= temp_qty

                    if open_side == OrderSide.SELL:
                        calc_tmp_profit += (ep / real_tp - self.trading_fee - 1) * temp_qty
                        # calc_tmp_profit += ep / real_tp - self.trading_fee
                    else:
                        calc_tmp_profit += (real_tp / ep - self.trading_fee - 1) * temp_qty

                    print("real_tp :", real_tp)
                    print("ep :", ep)
                    print("self.trading_fee :", self.trading_fee)
                    print("temp_qty :", temp_qty)

                    #            save exit data         #
                    #            exit_timeindex use "-1" index           #
                    # exit_timeindex = str(res_df.index[-1])
                    # trade_log[exit_timeindex] = [real_tp, "close"]
                    trade_log[k_ts] = [real_tp, "close"]

                with open(dir_path + "/trade_log/" + logger_name, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    print("exit trade_log dumped !")

                end_timestamp = int(time.time() * 1000)

                #           get total income from this trade       #
                while 1:
                    try:
                        income = total_income(self.symbol, start_timestamp, end_timestamp)
                        self.accumulated_income += income

                    except Exception as e:
                        print('error in total_income :', e)

                    else:
                        break

                tmp_profit = income / available_balance
                self.accumulated_profit *= (1 + tmp_profit)
                self.calc_accumulated_profit *= 1 + (calc_tmp_profit - 1) * leverage
                print('temporary profit : %.3f (%.3f) %%' % (tmp_profit * 100, (calc_tmp_profit - 1) * leverage * 100))
                print('accumulated profit : %.3f (%.3f) %%' % (
                    (self.accumulated_profit - 1) * 100, (self.calc_accumulated_profit - 1) * 100))
                print('accumulated income :', self.accumulated_income, 'USDT')
                print()
