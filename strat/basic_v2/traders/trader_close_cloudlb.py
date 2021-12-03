import os

# switch_path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend"
# os.chdir(switch_path)
os.chdir("../../../")
# print("os.getcwd() :", os.getcwd())
# quit()

from binance_futures_modules import *
from binance_futures_concat_candlestick import concat_candlestick
from easydict import EasyDict
from strat.basic_v1.utils_v2 import *
import pickle


class Trader:

    def __init__(self, symbol, interval, interval2, leverage_, initial_asset, stack_df=False, stacked_df_on=True):

        #       basic param     #
        self.last_index = -1
        self.symbol = symbol
        self.interval = interval
        self.interval2 = interval2
        self.use_rows = 200  # <-- important var. for "Trader" latency
        self.use_rows2 = 100
        # self.threshold = threshold
        # self.tp_gap = tp_gap
        self.leverage = leverage_
        self.initial_asset = initial_asset
        self.stack_df = stack_df
        self.stacked_df_on = stacked_df_on
        self.over_balance = None
        self.min_balance = 5.0  # USDT

        #       const param      #
        self.close_shift_size = 1
        self.cloud_shift_size = 1
        self.cloud_lookback = 69
        self.gap = 0.00005

        #       partial param       #
        self.partial_num = 1
        self.partial_qty_divider = 1.5

        #       pr param        #
        self.trading_fee = 0.0004
        self.accumulated_income = 0.0
        self.calc_accumulated_profit = 1.0
        self.accumulated_profit = 1.0
        self.accumulated_back_profit = 1.0

        sub_client.subscribe_aggregate_trade_event(symbol.lower(), callback, error)
        self.sub_client = sub_client

    def run(self):

        df_path = './basic_v1/%s.xlsx' % self.symbol

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
            with open('./basic_v1/config.json', 'r') as cfg:
                config = EasyDict(json.load(cfg))

            fundamental = config.FUNDMTL
            order = config.ORDER

            #       check run        #
            if not fundamental.run:
                continue

            #       init order side      #
            open_side = None
            close_side = None

            #       get startTime     #
            start_timestamp = int(time.time() * 1000)
            entry_const_check = True
            open_order = False
            # load_new_df = False

            short_open = False
            long_open = False
            while 1:

                if entry_const_check:

                    #       log last trading time     #
                    #       미체결을 고려해, entry_const_check 마다 log 수행       #
                    trade_log["last_trading_time"] = str(datetime.now())
                    with open("./basic_v1/trade_log/" + logger_name, "wb") as dict_f:
                        pickle.dump(trade_log, dict_f)

                    temp_time = time.time()

                    if self.stacked_df_on:

                        try:

                            new_df_, _ = concat_candlestick(self.symbol, self.interval, days=1, timesleep=0.2)

                            #       realtime candlestick confirmation       #
                            #       new_df 의 index 가 올바르지 않으면, load_new_df      #
                            if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                # load_new_df = True
                                continue

                            new_df2_, _ = concat_candlestick(self.symbol, self.interval2, days=1, timesleep=0.2)

                            #       new_df2 도 검수        #
                            if datetime.timestamp(new_df2_.index[-1]) < datetime.now().timestamp():
                                continue

                            new_df = new_df_.iloc[-self.use_rows:].copy()
                            new_df2 = new_df2_.iloc[-self.use_rows2:].copy()
                            # load_new_df = False
                            print("complete load_df execution time :", datetime.now())

                        except Exception as e:
                            print("error in load new_dfs :", e)
                            # load_new_df = True
                            continue

                        if self.stack_df:

                            try:
                                stacked_df = pd.read_excel(df_path, index_col=0)

                                #    use rows   #
                                #       add ep columns for append      #
                                # new_df['ep'] = np.nan
                                prev_complete_df = stacked_df.iloc[:-1, :]
                                stacked_df = prev_complete_df.append(new_df)

                                stacked_df = stacked_df[~stacked_df.index.duplicated(keep='first')].iloc[
                                             -order.use_rows:, :]

                            except Exception as e:
                                print("error in making stacked_df :", e)
                                continue

                    if self.stack_df:

                        try:
                            if not self.stacked_df_on:
                                days = calc_train_days(interval=self.interval, use_rows=order.use_rows)
                                stacked_df, _ = concat_candlestick(self.symbol, self.interval, days=days, timesleep=0.2)
                                stacked_df = stacked_df.iloc[-order.use_rows:, :]
                                self.stacked_df_on = True

                            stacked_df.to_excel(df_path)

                        except Exception as e:
                            print("error in load new stacked_df :", e)
                            self.stacked_df_on = False
                            continue

                    print("~ load_new_df time : %.2f" % (time.time() - temp_time))
                    # temp_time = time.time()

                    try:
                        #       latency 를 줄이기 위해서는, input df length 를 조절해야함     #
                        res_df = sync_check(new_df, new_df2, cloud_on=True)

                        print('~ sync_check time : %.5f' % (time.time() - temp_time))

                    except Exception as e:
                        print("error in sync_check :", e)
                        continue

                    #       1. open order constraints phase      #
                    try:
                        cloud_top = np.max(res_df[["senkou_a1", "senkou_b1"]], axis=1)
                        cloud_bottom = np.min(res_df[["senkou_a1", "senkou_b1"]], axis=1)

                        #        1. check cloud_top data type     #
                        # print("cloud_top :", cloud_top)
                        # quit()

                        upper_ep = res_df['min_upper'] * (1 - self.gap)
                        lower_ep = res_df['max_lower'] * (1 + self.gap)

                        # print('1')

                        under_top = res_df['close'].shift(self.close_shift_size) <= cloud_top.shift(self.cloud_shift_size)
                        over_bottom = res_df['close'].shift(self.close_shift_size) >= cloud_bottom.shift(self.cloud_shift_size)

                        # print('2')
                        # quit()

                        #       Todo        #
                        #        save res_df before, entry const check      #
                        excel_name = str(datetime.now()).replace(":", "").split(".")[0]
                        res_df.to_excel("./basic_v1/df_log/%s.xlsx" % excel_name)

                        #       3. res_df 의 모든 length 에 대해 할 필요는 없을 것       #
                        #       3.1 양방향에 limit 을 걸어두고 체결되면 미체결 주문은 all cancel       #
                        #       3.2 cloud lb const 적용하면, 양방향 될일 거의 없지 않을까, 그래도 양방향 코드 진행       #
                        #       a. st const        #
                        if np.sum(res_df[['minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend']].iloc[
                                      self.last_index]) != 3:

                            #     b. check cloud constraints   #
                            if np.sum(under_top.iloc[-self.cloud_lookback:]) == self.cloud_lookback:
                                short_open = True
                                # print("short_open :", short_open)
                                # print("sum trend :", res_df[['minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend']].iloc[self.last_index])
                                # print("under_top :", under_top.iloc[-self.cloud_lookback:])

                        if np.sum(res_df[['minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend']].iloc[
                                      self.last_index]) != -3:

                            #     b. check cloud constraints   #
                            if np.sum(over_bottom.iloc[-self.cloud_lookback:]) == self.cloud_lookback:
                                long_open = True
                                # print("long_open :", long_open)
                                # print("sum trend :", res_df[['minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend']].iloc[self.last_index])
                                # print("over_bottom :", over_bottom.iloc[-self.cloud_lookback:])

                    except Exception as e:
                        print("error in open const phase :", e)
                        continue

                    entry_const_check = False

                    if short_open or long_open:
                        pass
                    else:
                        continue

                    #         enlist ep         #
                    try:
                        df = enlist_epindf(res_df, upper_ep, lower_ep)
                        print('res_df.index[-1] :', res_df.index[-1])
                        print('~ ep enlist time : %.5f' % (time.time() - temp_time))
                        print()
                        open_order = True

                    except Exception as e:
                        print('error in enlist_epindf :', e)
                        entry_const_check = True    # res_df 에 data 가 덮여씌워질 것임
                        continue

                if open_order:

                    #            1. entry const 에 따른, 단방향 limit 임          #
                    #            2. code 는 양방향 limit 으로 진행               #

                    #           이건 단방향, 일단은 단방향으로 진행, 차후 양방향으로 변경 가능          #
                    if short_open:
                        open_side = OrderSide.SELL
                    elif long_open:
                        open_side = OrderSide.BUY

                    #           양방향은 밑에서, open_order 을 두번 하면댐       #
                    #           (open order param list 를 만들어야겠지         #

                    # if open_side is not None:
                    #      if open_order = True, just break     #
                    break

                time.sleep(fundamental.realtime_term)  # <-- term for realtime market function

                #       check tick change      #
                #       res_df 마지막 timestamp index 보다 current timestamp 이 커지면 load_new_df       #

                if datetime.timestamp(res_df.index[-1]) < datetime.now().timestamp():
                    entry_const_check = True
                    # load_new_df = True
                    # open_order = False
                    print('res_df[-1] timestamp :', datetime.timestamp(res_df.index[-1]))
                    print('current timestamp :', datetime.now().timestamp())
                    print()

                    # time.sleep(fundamental.close_complete_term)   # <-- term for close completion

            # print('realtime_price :', realtime_price)

            first_iter = True
            while 1:  # <-- loop for 'check order type change condition'

                print('open_side :', open_side)

                #             load open order variables              #
                if order.entry_type == OrderType.MARKET:
                    #             ep with market            #
                    try:
                        ep = get_market_price_v2(self.sub_client)
                    except Exception as e:
                        print('error in get_market_price :', e)
                else:
                    #             ep with Limit             #
                    if open_side == OrderSide.BUY:
                        ep = df['long_ep'].iloc[self.last_index]
                        str_open_side = "long"

                    else:
                        ep = df['short_ep'].iloc[self.last_index]
                        str_open_side = "short"

                #           tp & sl             #
                sl = df['sl'].iloc[self.last_index]
                # tp = df['tp'].iloc[self.last_index]
                # leverage = df['leverage'].iloc[self.last_index]  # <-- 왜 leverage 를 df 로부터 다시 가져와야하는지 모르겠음
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
                sl = calc_with_precision(sl, price_precision)
                leverage = min(limit_leverage, leverage)
                print('ep :', ep)
                print('sl :', sl)
                print('leverage :', leverage)
                print('~ ep sl lvrg set time : %.5f' % (time.time() - temp_time))

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

                if order.entry_type == OrderType.MARKET:

                    while 1:  # <-- loop for complete open order
                        try:
                            #           market order            #
                            request_client.post_order(symbol=self.symbol, side=open_side, ordertype=OrderType.MARKET,
                                                      quantity=str(open_quantity))
                        except Exception as e:
                            print('error in open order :', e)

                            open_retry_cnt += 1
                            if 'insufficient' in str(e):

                                try:
                                    max_available_balance = get_availableBalance()
                                    #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                                    self.over_balance = available_balance
                                    available_balance = max_available_balance * 0.9
                                    #          Get available open_quantity         #
                                    open_quantity = available_balance / ep * leverage
                                    open_quantity = calc_with_precision(open_quantity, quantity_precision)
                                    print('available_balance (temp) :', available_balance)
                                    print('open_quantity :', open_quantity)

                                except Exception as e:
                                    print('error in get_availableBalance() :', e)

                            elif open_retry_cnt > 100:
                                print('open_retry_cnt over 100')
                                quit()
                            continue
                        else:
                            print('open order enlisted :', datetime.now())
                            break

                    #       Enough time for open_quantity Consuming      #
                    time.sleep(60 - fundamental.bar_close_second)

                else:

                    while 1:  # <-- loop for complete open order
                        #       If Limit Order used, Set Order Execute Time & Check Remaining Order         #
                        try:
                            #           limit order            #
                            result = request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                                               side=open_side,
                                                               ordertype=order.entry_type,
                                                               quantity=str(open_quantity), price=str(ep),
                                                               reduceOnly=False)
                        except Exception as e:
                            print('error in open order :', e)

                            open_retry_cnt += 1
                            if 'insufficient' in str(e):

                                try:
                                    max_available_balance = get_availableBalance()
                                    #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                                    self.over_balance = available_balance
                                    available_balance = max_available_balance * 0.9
                                    #          Get available open_quantity         #
                                    open_quantity = available_balance / ep * leverage
                                    open_quantity = calc_with_precision(open_quantity, quantity_precision)
                                    print('available_balance (temp) :', available_balance)
                                    print('open_quantity :', open_quantity)

                                except Exception as e:
                                    print('error in get_availableBalance() :', e)

                            elif open_retry_cnt > 100:
                                print('open_retry_cnt over 100')
                                quit()
                            continue
                        else:
                            print('open order enlisted :', datetime.now())
                            break

                    #       set order execute time & check breakout_qty_ratio         #
                    while 1:

                        #       해당 종가까지 체결 대기       #
                        #       datetime.timestamp(res_df.index[-1]) -> 2분이면, 01:59:999 이런식으로 될것
                        if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                            # if datetime.now().timestamp() + order.entry_execution_wait > datetime.timestamp(res_df.index[-1]):
                            break

                        time.sleep(fundamental.realtime_term)  # <-- for realtime price function

                #           포지션에 독립적으로, 미체결 open order 모두 cancel               #
                #           regardless to position exist, cancel all open orders       #
                try:
                    result = request_client.cancel_all_orders(symbol=self.symbol)
                    print()
                except Exception as e:
                    print('error in cancel remaining open order :', e)

                if orderside_changed:
                    first_iter = False
                    print('orderside_changed :', orderside_changed)
                    print()
                    continue
                else:
                    break

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
                #        2. check str(res_df.index[-1])     #
                entry_timeindex = str(res_df.index[-1])
                trade_log[entry_timeindex] = [ep, str_open_side, "open"]

                with open("./basic_v1/trade_log/" + logger_name, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    print("entry trade_log dumped !")

                print()

                #           set close side          #
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                #           market : prevent close at open bar           #
                if order.entry_type == OrderType.MARKET:
                    time.sleep(60 - datetime.now().second)

                tp = None
                load_new_df2 = True
                while 1:

                    #       Todo        #
                    #        0. websocket 으로 load_new_df 가능할지 확인             #

                    #        1-1. load new_df in every new_df2 index change           #
                    #             -> 이전 index 와 비교하려면, 어차피 load_new_df 해야함      #
                    #        2. reorder tp 경우, enlist 된 기존 tp cancel 해야함
                    #        2-1. new_df check 동일하게 수행함

                    if load_new_df2:

                        try:
                            new_df_, _ = concat_candlestick(self.symbol, self.interval, days=1, timesleep=0.2)

                            if datetime.timestamp(new_df_.index[-1]) < datetime.now().timestamp():
                                continue

                            new_df2_, _ = concat_candlestick(self.symbol, self.interval2, days=1, timesleep=0.2)

                            if datetime.timestamp(new_df2_.index[-1]) < datetime.now().timestamp():
                                continue

                            new_df = new_df_.iloc[-self.use_rows:].copy()
                            new_df2 = new_df2_.iloc[-self.use_rows2:].copy()
                            res_df = sync_check(new_df, new_df2)

                            prev_tp = tp

                            if open_side == OrderSide.SELL:
                                tp = res_df['middle_line'].iloc[self.last_index] * (1 + self.gap)
                            else:
                                tp = res_df['middle_line'].iloc[self.last_index] * (1 - self.gap)

                            print("tp :", tp)

                        except Exception as e:
                            print('error in get new_dfs (tp phase) :', e)
                            continue

                        else:
                            load_new_df2 = False

                    #         1. tp reorder 는 tp 변동이 있는 경우만      #
                    #         2. initial_tp 는 무조건 order 진행         #
                    #         3. reorder 의 경우, 기존 tp order cancel 해야함       #
                    if prev_tp is None or tp != prev_tp:

                        if tp != prev_tp:

                            #           미체결 tp order 모두 cancel               #
                            #           regardless to position exist, cancel all tp orders       #
                            try:
                                result = request_client.cancel_all_orders(symbol=self.symbol)
                            except Exception as e:
                                print('error in cancel remaining tp order :', e)
                                continue

                        #         tp order enlist phase         #
                        #         get price, volume precision -> reatlime 가격 변동으로 인한 precision 변동 가능성       #
                        try:
                            price_precision, quantity_precision = get_precision(self.symbol)
                        except Exception as e:
                            print('error in get price & volume precision :', e)
                            continue
                        else:
                            print('price_precision :', price_precision)
                            print('quantity_precision :', quantity_precision)

                        #           Todo            #
                        #            1. partial 을 ep 기준으로 하면 안댐         #
                        #            ==> dynamic tp 기 때문에       #
                        #            2. 따라서, 일단은 tp_list 를 직접 만들어준다     #
                        #            2-1. tp_list 는 ep 로부터 먼 tp 부터 정렬한다     #
                        # tp_list = get_partial_tplist(ep, tp, open_side, self.partial_num, price_precision)
                        tp_list = [tp]
                        tp_list = list(map(lambda x: calc_with_precision(x, price_precision), tp_list))

                        print('tp_list :', tp_list)

                        try:
                            partial_limit(self.symbol, tp_list, close_side, quantity_precision, self.partial_qty_divider)

                        except Exception as e:
                            print('error in partial_limit :', e)
                            time.sleep(1)
                            continue

                        else:
                            print("tp order enlisted :", datetime.now())

                    #            1. 매 1분동안 체결 검사 진행                 #
                    tp_on, sl_on = True, False
                    while 1:

                        # #           sl condition phase            #
                        # #           wait until final countdown before closing (55 second)      #
                        # current_datetime = datetime.now()
                        # if current_datetime.second >= (fundamental.bar_close_second - 4):
                        #
                        #     #          close by realtime price        #
                        #     try:
                        #         realtime_price = get_market_price_v2(self.sub_client)
                        #
                        #     except Exception as e:
                        #         print('error in get_market_price :', e)
                        #         continue
                        #
                        #     print(current_datetime, 'realtime_price :', realtime_price)
                        #     sl_on = True

                        #        1. 이거 맞는거 같긴 한데, 제대로 작동하는지만 확인       #
                        try:
                            tp_remain_quantity = get_remaining_quantity(self.symbol)
                        except Exception as e:
                            print('error in get_remaining_quantity :', e)
                            continue

                        #       들고 있는 position quantity 가 0 이면, 이 거래를 끝임        #
                        if tp_remain_quantity == 0.0:
                            print('all tp order executed')
                            break

                        if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):
                            load_new_df2 = True
                            break

                    if load_new_df2:
                        continue

                    #   이 phase 를 오는 경우는, all tp 체결 또는 sl 조건 완성     #
                    if not sl_on:
                        break

                    #               sl phase               #
                    #                1. warning, sl 하려면, exit price 필요함         #
                    else:

                        remain_tp_canceled = False
                        while 1:  # <--- This loop for sl close & complete close

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
                                        result = request_client.cancel_all_orders(symbol=self.symbol)

                                    except Exception as e:
                                        print('error in cancel remaining tp order (sl phase) :', e)
                                        continue

                                remain_tp_canceled = True

                            #          get remaining sl_remain_quantity         #
                            try:
                                sl_remain_quantity = get_remaining_quantity(self.symbol)

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

                            sl_remain_quantity = calc_with_precision(sl_remain_quantity, quantity_precision)
                            print('sl_remain_quantity :', sl_remain_quantity)

                            #           close remaining sl_remain_quantity             #
                            #           1. order side should be opposite side to the open          #
                            #           2. reduceOnly = 'true'       #
                            try:
                                if not order.sl_type == OrderType.MARKET:

                                    #           stop limit order            #
                                    # if close_side == OrderSide.SELL:
                                    #     stop_price = calc_with_precision(realtime_price + 5 * 10 ** -price_precision,
                                    #                                      price_precision)
                                    # else:
                                    #     stop_price = calc_with_precision(realtime_price - 5 * 10 ** -price_precision,
                                    #                                      price_precision)

                                    # request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                    #                           side=close_side,
                                    #                           ordertype=order.sl_type, stopPrice=str(stop_price),
                                    #                           quantity=str(sl_remain_quantity), price=str(realtime_price),
                                    #                           reduceOnly=True)

                                    if pd.isna(sl):
                                        exit_price = realtime_price
                                    else:
                                        exit_price = sl

                                    #           limit order             #
                                    request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                                              side=close_side,
                                                              ordertype=order.sl_type,
                                                              quantity=str(sl_remain_quantity), price=str(exit_price),
                                                              reduceOnly=True)

                                else:
                                    #           market order            #
                                    result = request_client.post_order(symbol=self.symbol, side=close_side,
                                                                       ordertype=OrderType.MARKET,
                                                                       quantity=str(sl_remain_quantity),
                                                                       reduceOnly=True)

                            except Exception as e:
                                print('error in close order :', e)

                                #       Check error msg     #
                                if 'Quantity less than zero' in str(e):
                                    break

                                order.sl_type = OrderType.MARKET
                                continue
                            else:
                                print('sl order enlisted')

                            #       enough time for sl_remain_quantity to be consumed      #
                            if not order.sl_type == OrderType.MARKET:

                                #       해당 틱의 종가까지 기다림      #
                                time.sleep(order.exit_execution_wait - datetime.now().second)
                                print("order.exit_execution_wait - datetime.now().second :",
                                      order.exit_execution_wait - datetime.now().second)
                                print("datetime.now().second :", datetime.now().second)

                            else:
                                time.sleep(1)

                            #               check remaining sl_remain_quantity             #
                            try:
                                sl_remain_quantity = get_remaining_quantity(self.symbol)
                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                continue

                            if sl_remain_quantity == 0.0:
                                print('sl order executed')
                                break

                            else:
                                #           complete close by market            #
                                print('sl_type changed to market')
                                order.sl_type = OrderType.MARKET
                                continue

                        break  # <--- break for all close order loop, break partial tp loop

                #       check back-test profit     #
                while 1:
                    if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]):

                        # if datetime.now().timestamp() > datetime.timestamp(res_df.index[-1]) + \
                        #         fundamental.close_complete_term:
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

                #        calc logical profit      #
                #        check, limit -> market condition        #

                #             real_tp, division           #
                #             prev_tp == tp 인 경우는, tp > open 인 경우에도 real_tp 는 tp 와 동일함      #
                if close_side == OrderSide.BUY:

                    if prev_tp == tp:
                        real_tp = tp

                    else:
                        if tp > res_df['open'].iloc[-1]:
                            real_tp = res_df['open'].iloc[-1]
                            print("market tp executed !")
                        else:
                            real_tp = tp

                    calc_tmp_profit = ep / real_tp - self.trading_fee

                else:

                    if prev_tp == tp:
                        real_tp = tp

                    else:
                        if tp < res_df['open'].iloc[-1]:
                            real_tp = res_df['open'].iloc[-1]
                            print("market tp executed !")
                        else:
                            real_tp = tp

                    calc_tmp_profit = real_tp / ep - self.trading_fee

                #           Todo            #
                #            check, tp & ep             #
                print("real_tp :", real_tp)
                print("ep :", ep)
                print("self.trading_fee :", self.trading_fee)

                #            save exit data         #
                #            exit_timeindex use "-1" index           #
                exit_timeindex = str(res_df.index[-1])
                trade_log[exit_timeindex] = [real_tp, "close"]

                with open("./basic_v1/trade_log/" + logger_name, "wb") as dict_f:
                    pickle.dump(trade_log, dict_f)
                    print("exit trade_log dumped !")

                end_timestamp = int(time.time() * 1000)

                #           get total income from this trade       #
                try:
                    income = total_income(self.symbol, start_timestamp, end_timestamp)
                    self.accumulated_income += income

                except Exception as e:
                    print('error in total_income :', e)

                tmp_profit = income / available_balance
                self.accumulated_profit *= (1 + tmp_profit)
                self.calc_accumulated_profit *= 1 + (calc_tmp_profit - 1) * leverage
                print('temporary profit : %.3f (%.3f) %%' % (tmp_profit * 100, (calc_tmp_profit - 1) * leverage * 100))
                print('accumulated profit : %.3f (%.3f) %%' % (
                    (self.accumulated_profit - 1) * 100, (self.calc_accumulated_profit - 1) * 100))
                print('accumulated income :', self.accumulated_income, 'USDT')
                print()
