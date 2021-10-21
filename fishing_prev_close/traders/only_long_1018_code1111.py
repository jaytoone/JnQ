import os
# switch_path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend"
# os.chdir(switch_path)
os.chdir("../../")

from binance_funcs.binance_futures_modules import *
from funcs.funcs_for_trade import *
from binance_funcs.binance_futures_concat_candlestick import concat_candlestick
from easydict import EasyDict
from fishing_prev_close.utils import arima_profit, calc_train_days, tp_update, interval_to_min


class ARIMA_Bot:

    def __init__(self, symbol, interval, tp, leverage_, initial_asset, stacked_df_on=False):

        self.last_index = -1
        self.symbol = symbol
        self.interval = interval
        # self.threshold = threshold
        self.tp, self.leverage_ = tp, leverage_
        self.initial_asset = initial_asset
        self.stacked_df_on = stacked_df_on
        self.over_balance = None
        self.min_balance = 5.0  # USDT

        self.trading_fee = 0.0006
        self.accumulated_income = 0.0
        self.calc_accumulated_profit = 1.0
        self.accumulated_profit = 1.0
        self.accumulated_back_profit = 1.0

        sub_client.subscribe_aggregate_trade_event(symbol.lower(), callback, error)
        self.sub_client = sub_client

    def run(self):

        df_path = './fishing_prev_close/%s.xlsx' % self.symbol

        #         1. Leverage type => Isolated          #
        try:
            request_client.change_margin_type(symbol=self.symbol, marginType=FuturesMarginType.ISOLATED)
        except Exception as e:
            print('error in change_margin_type :', e)
        else:
            print('Leverage type --> Isolated')

        #         2. Confirm Limit Leverage          #
        try:
            limit_leverage = get_limit_leverage(symbol_=self.symbol)
        except Exception as e:
            print('error in get_limit_leverage :', e)
            quit()
        else:
            print('Limit Leverage :', limit_leverage)

        print()
        print('Trade Running Now...')

        while 1:

            #       Configuration       #
            with open('./fishing_prev_close/config.json', 'r') as cfg:
                config = EasyDict(json.load(cfg))

            fundamental = config.FUNDMTL
            order = config.ORDER

            #       Run Decision        #
            if not fundamental.run:
                continue

            #       Init Order side      #
            open_side = None
            close_side = None

            #       Get startTime for Total Income calc    #
            start_timestamp = int(time.time() * 1000)
            fishing_survey = True
            arima_on = False
            get_first_df = False
            stack_df = False
            while 1:

                #       arima phase 도 ai survey pass 시 진행해야한다           #
                if fishing_survey:  # 해당 interval 내에 arima survey 수행을 한번만 하기 위한 조건 phase

                    try:
                        temp_time = time.time()

                        if self.stacked_df_on:

                            try:
                                back_df
                            except Exception as e:
                                get_first_df = True
                            else:
                                first_df = back_df

                            if get_first_df:
                                first_df, _ = concat_candlestick(self.symbol, self.interval, days=1, limit=order.use_rows,
                                                            timesleep=0.2, show_process=False)
                                get_first_df = False

                            #       realtime candlestick confirmation       #
                            if datetime.timestamp(first_df.index[-1]) < datetime.now().timestamp():
                                get_first_df = True
                                continue

                            if stack_df:
                                stacked_df = pd.read_excel(df_path, index_col=0)
                            else:
                                stacked_df = first_df

                            #    use rows   #
                            #       add ep columns for append      #
                            # first_df['ep'] = np.nan
                            prev_complete_df = stacked_df.iloc[:-1, :]
                            if stack_df:
                                stacked_df = prev_complete_df.append(first_df)
                                stacked_df = stacked_df[~stacked_df.index.duplicated(keep='first')].iloc[-order.use_rows:, :]

                            print("complete load_df execution time :", datetime.now())

                        if not self.stacked_df_on:
                            days = calc_train_days(interval=self.interval, use_rows=order.use_rows)
                            stacked_df, _ = concat_candlestick(self.symbol, self.interval, days=days, timesleep=0.2)
                            stacked_df = stacked_df.iloc[-order.use_rows:, :]
                            self.stacked_df_on = True

                        if stack_df:
                            stacked_df.to_excel(df_path)

                        #         fishing survey phase       #
                        fishing_survey = False

                        # ----------- set constraints for rapid drop ----------- #
                        prev_complete_df = stacked_df.iloc[:-1, :]
                        print("~ load_new_df time : %.2f" % (time.time() - temp_time))

                        # ma = prev_complete_df['close'].rolling(120).mean()
                        # print("ma.iloc[-2:].values :", ma.iloc[-2:].values)

                        # ema = prev_complete_df['close'].ewm(span=190, min_periods=190 - 1, adjust=False).mean()
                        # print("ema.iloc[-2:].values :", ema.iloc[-2:].values)
                        #
                        # if ema.iloc[-1] > ema.iloc[-2]:
                        #     pass
                        # else:
                        #     continue

                        #         ARIMA survey phase / please use complete data          #
                        df, pred_close, err_range = arima_profit(prev_complete_df, tp=self.tp, leverage=self.leverage_)
                        print('stacked_df.index[-1] :', stacked_df.index[-1],
                              'checking time : %.2f' % (time.time() - temp_time))
                        print('pred_close :', pred_close)
                        arima_on = True
                        print('~ tp set time : %.5f' % (time.time() - temp_time))

                    except Exception as e:
                        print('error in arima_profit :', e)
                        continue

                if arima_on:

                    #           Continuously check open condition          #
                    #      Get realtime market price & compare with trigger price      #
                    # try:
                    #     realtime_price = get_market_price_v2(self.sub_client)
                    #
                    # except Exception as e:
                    #     print('error in get_market_price :', e)
                    #     continue

                    # if realtime_price < pred_close - err_range / 2:
                    #     open_side = OrderSide.BUY

                    open_side = OrderSide.BUY

                    #       if arima_on is True, just enlist open_order     #
                    # if np.argmax(test_result, axis=1)[-1] == 1:
                    #     open_side = OrderSide.BUY
                    # else:
                    #     open_side = OrderSide.SELL

                    if open_side is not None:
                        break

                time.sleep(fundamental.realtime_term)  # <-- term for realtime market function

                #       check tick change      #
                #       stacked_df의 마지막 timestamp index 보다 current timestamp 이 커지면 init arima       #
                if datetime.timestamp(stacked_df.index[-1]) < datetime.now().timestamp():
                    fishing_survey = True
                    arima_on = False
                    print('stacked_df[-1] timestamp :', datetime.timestamp(stacked_df.index[-1]))  # <-- code proof
                    print('current timestamp :', datetime.now().timestamp())

                    time.sleep(fundamental.close_complete_term)   # <-- term for close completion

            # print('realtime_price :', realtime_price)
            print('We Got Open Signal.')

            first_iter = True
            while 1:  # <-- loop for 'check order type change condition'

                print('open_side :', open_side)

                #             Load Trade Options              #
                if order.entry_type == OrderType.MARKET:
                    #             ep with Market            #
                    try:
                        ep = get_market_price_v2(self.sub_client)
                    except Exception as e:
                        print('error in get_market_price :', e)
                else:
                    #             ep with Limit             #
                    if open_side == OrderSide.BUY:
                        ep = df['long_ep'].iloc[self.last_index]

                    else:
                        ep = df['short_ep'].iloc[self.last_index]

                #           TP & SL             #
                sl_level = df['sl_level'].iloc[self.last_index]
                tp_level = df['tp_level'].iloc[self.last_index]
                leverage = df['leverage'].iloc[self.last_index]

                #         Get price, volume precision --> 가격 변동으로 인한 precision 변동 가능성       #
                try:
                    price_precision, quantity_precision = get_precision(self.symbol)
                except Exception as e:
                    print('error in get price & volume precision :', e)
                    continue
                else:
                    print('price_precision :', price_precision)
                    print('quantity_precision :', quantity_precision)

                ep = calc_with_precision(ep, price_precision)
                sl_level = calc_with_precision(sl_level, price_precision)
                leverage = min(limit_leverage, leverage)
                print('ep :', ep)
                print('sl_level :', sl_level)
                print('leverage :', leverage)
                print('~ ep sl lvrg set time : %.5f' % (time.time() - temp_time))

                partial_num = 1
                partial_qty_divider = 1.5

                if not pd.isna(tp_level):

                    tp_list = list()
                    if open_side == OrderSide.BUY:

                        for part_i in range(partial_num, 0, -1):
                            tmp_tp_level = ep + abs(tp_level - ep) * (part_i / partial_num)
                            tp_list.append(tmp_tp_level)

                        if order.entry_type == OrderType.MARKET:
                            if ep <= sl_level:
                                print('ep <= level : %s <= %s' % (ep, sl_level))
                                continue  # while loop 가 위에 형성되면서 차후 사용을 위해선 break 으로 바뀌어야할 것
                    else:

                        for part_i in range(partial_num, 0, -1):
                            tmp_tp_level = ep - abs(tp_level - ep) * (part_i / partial_num)
                            tp_list.append(tmp_tp_level)

                        if order.entry_type == OrderType.MARKET:
                            if ep >= sl_level:
                                print('ep >= level : %s >= %s' % (ep, sl_level))
                                continue

                    tp_list = list(map(lambda x: calc_with_precision(x, price_precision), tp_list))
                    print('tp_list :', tp_list)

                try:
                    request_client.change_initial_leverage(symbol=self.symbol, leverage=leverage)
                except Exception as e:
                    print('error in change_initial_leverage :', e)
                else:
                    print('leverage changed -->', leverage)

                if first_iter:
                    #          Define Start Asset          #
                    if self.accumulated_income == 0.0:

                        available_balance = self.initial_asset  # USDT

                    else:
                        available_balance += income

                #          Get availableBalance          #
                try:
                    max_available_balance = get_availableBalance()

                    #       over_balance 가 저장되어 있다면, 지속적으로 max_balance 와의 비교 진행      #
                    if self.over_balance is not None:
                        if self.over_balance <= max_available_balance * 0.9:
                            available_balance = self.over_balance
                            self.over_balance = None
                        else:
                            available_balance = max_available_balance * 0.9  # over_balance 를 넘지 않는 선에서 max_balance 채택

                    else:   # <-- 예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치
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

                #          Get available quantity         #
                quantity = available_balance / ep * leverage
                quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')

                if self.over_balance is not None:
                    print('available_balance (temp) :', available_balance)
                else:
                    print('available_balance :', available_balance)

                print('quantity :', quantity)

                #           Open            #
                orderside_changed = False
                open_retry_cnt = 0

                if order.entry_type == OrderType.MARKET:

                    while 1:    # <-- loop for complete open order
                        try:
                            #           Market Order            #
                            request_client.post_order(symbol=self.symbol, side=open_side, ordertype=OrderType.MARKET,
                                                      quantity=str(quantity))
                        except Exception as e:
                            print('error in Open Order :', e)

                            open_retry_cnt += 1

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
                            print('Open order listed.')
                            break

                    #       Enough time for quantity Consuming      #
                    time.sleep(60 - fundamental.bar_close_second)

                else:

                    code_1111 = 0
                    while 1:    # <-- loop for complete open order
                        #       If Limit Order used, Set Order Execute Time & Check Remaining Order         #
                        try:
                            #           Limit Order            #
                            request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                                               side=open_side,
                                                               ordertype=order.entry_type,
                                                               quantity=str(quantity), price=str(ep),
                                                               reduceOnly=False)
                        except Exception as e:
                            print('error in Open Order :', e)

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
                                code_1111 = 1
                                print("ep :", ep)
                                print("quantity :", quantity)
                                print()
                                break

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
                            print('Open order listed.')
                            break

                    #       Set Order Execute Time & Check breakout_qty_ratio         #
                    while 1:

                        if code_1111:
                            break

                        #       wait interval ends     #
                        if datetime.now().timestamp() + order.entry_execution_wait > datetime.timestamp(stacked_df.index[-1]):
                            break

                        #           Check order type change condition           #
                        #      Get realtime market price & compare with trigger price      #
                        # try:
                        #     realtime_price = get_market_price_v2(self.sub_client)
                        # except Exception as e:
                        #     print('error in get_market_price :', e)
                        #     continue

                        #          Check order type change condition       #
                        # if open_side == OrderSide.SELL:
                        #     if realtime_price < pred_close - err_range:
                        #
                        #         #       체결 내역이 존재하지 않으면,        # <-- api limit 을 고려해 내부로 가져옴
                        #         try:
                        #             exec_quantity = get_remaining_quantity(self.symbol)
                        #         except Exception as e:
                        #             print('error in exec_quantity check :', e)
                        #             continue
                        #
                        #         if exec_quantity == 0.0:
                        #             open_side = OrderSide.BUY
                        #             orderside_changed = True
                        #             break
                        #
                        # else:
                        #     if realtime_price > pred_close + err_range:
                        #
                        #         #       체결 내역이 존재하지 않으면,        #
                        #         try:
                        #             exec_quantity = get_remaining_quantity(self.symbol)
                        #         except Exception as e:
                        #             print('error in exec_quantity check :', e)
                        #             continue
                        #
                        #         if exec_quantity == 0.0:
                        #             open_side = OrderSide.SELL
                        #             orderside_changed = True
                        #             break

                        time.sleep(fundamental.realtime_term)  # <-- for realtime price function

                #           Remaining Order check            #
                # remained_orderId = None
                #
                # try:
                #     remained_orderId = remaining_order_check(self.symbol)
                # except Exception as e:
                #     print('error in remaining_order_check :', e)

                # if remained_orderId is not None:
                #           regardless to position exist, cancel all open orders       #
                try:
                    request_client.cancel_all_orders(symbol=self.symbol)
                except Exception as e:
                    print('error in cancel remaining open order :', e)

                if orderside_changed:
                    first_iter = False
                    print('orderside_changed :', orderside_changed)
                    print()
                    continue
                else:
                    break

            try:
                if available_balance < self.min_balance:
                    continue
            except Exception as e:
                pass

            while 1:
                try:
                    exec_quantity = get_remaining_quantity(self.symbol)
                except Exception as e:
                    print('error in exec_quantity check :', e)
                    continue
                break

            if exec_quantity == 0.0:

                while 1:

                    if code_1111:
                        break

                    #        체결량 없을시 연속적인 open order 를 방지해야한다        #
                    if datetime.now().timestamp() > datetime.timestamp(stacked_df.index[-1]) + fundamental.close_complete_term:  # close 데이터가 완성되기 위해 충분한 시간

                        #       Check back-tested_Profit     #
                        while 1:
                            try:
                                back_df, _ = concat_candlestick(self.symbol, self.interval, days=1, limit=order.use_rows,
                                                            timesleep=0.2, show_process=False)

                                #       realtime candlestick confirmation       #
                                if datetime.timestamp(back_df.index[-1]) > datetime.now().timestamp():
                                    break

                            except Exception as e:
                                print('error in back-test_Profit :', e)

                        #       init temporary back profit      #
                        back_tmp_profit = 1.0

                        if open_side == OrderSide.BUY:
                            if back_df['low'].iloc[-2] < df['long_ep'].iloc[self.last_index]:
                                back_tmp_profit = back_df['close'].iloc[-2] / df['long_ep'].iloc[self.last_index] - self.trading_fee
                        else:
                            if back_df['high'].iloc[-2] > df['short_ep'].iloc[self.last_index]:
                                back_tmp_profit = df['short_ep'].iloc[self.last_index] / back_df['close'].iloc[-2] - self.trading_fee

                        self.accumulated_back_profit *= 1 + (back_tmp_profit - 1) * leverage
                        print('temporary_back_Profit : %.3f %%' % ((back_tmp_profit - 1) * leverage * 100))
                        # print('accumulated_back_Profit : %.3f %%' % ((self.accumulated_back_profit - 1) * 100))
                        print()
                        break

                income = 0
                # continue

            #       position / 체결량 이 존재하면 close order 진행      #
            else:

                print('Open order executed.')
                print()

                #           side Change         #
                if open_side == OrderSide.BUY:
                    close_side = OrderSide.SELL
                else:
                    close_side = OrderSide.BUY

                #           Market : Prevent close at open bar           #
                if order.entry_type == OrderType.MARKET:
                    time.sleep(60 - datetime.now().second)

                while 1:  # TP limit close loop

                    if not pd.isna(tp_level):

                        try:
                            partial_limit(self.symbol, tp_list, close_side, quantity_precision, partial_qty_divider)

                        except Exception as e:
                            print('error in partial_limit :', e)
                            time.sleep(1)
                            continue

                    tp_switch, sl_switch = False, False  # tp_switch 는 limit_close 의 경우에 사용한다.
                    while 1:

                        current_datetime = datetime.now()

                        #           Wait until final countdown before closing (55 second)      #
                        if current_datetime.second >= (fundamental.bar_close_second - 4):

                            #          Close by realtime price        #
                            try:
                                realtime_price = get_market_price_v2(self.sub_client)

                            except Exception as e:
                                print('error in get_market_price :', e)
                                continue

                            print(current_datetime, 'realtime_price :', realtime_price)
                            sl_switch = True

                        if tp_switch or sl_switch:
                            break

                    print('We Got Close Signal.')

                    if tp_switch and not pd.isna(tp_level):  # Trade Done!
                        break
                    else:
                        cancel_tp = False
                        while 1:  # <--- This Loop for SL Close & Re-Close

                            #               Canceling Limit Order Once               #
                            if not cancel_tp:
                                #               Remaining TP close Order check            #
                                try:
                                    remained_orderId = remaining_order_check(self.symbol)
                                except Exception as e:
                                    print('error in remaining_order_check :', e)
                                    continue

                                if remained_orderId is not None:
                                    #           If Remained position exist, Cancel it       #
                                    try:
                                        result = request_client.cancel_all_orders(symbol=self.symbol)
                                    except Exception as e:
                                        print('error in cancel remaining TP Close order :', e)
                                        continue

                                cancel_tp = True

                            #          Get Remaining quantity         #
                            try:
                                quantity = get_remaining_quantity(self.symbol)
                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                continue

                            #           Get price, volume precision       #
                            try:
                                _, quantity_precision = get_precision(self.symbol)
                            except Exception as e:
                                print('error in get price & volume precision :', e)
                            else:
                                print('quantity_precision :', quantity_precision)

                            quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')
                            print('quantity :', quantity)

                            #           Close             #
                            #           1. Order side should be Opposite side to the Open          #
                            #           2. ReduceOnly = 'true'       #
                            try:
                                if not order.sl_type == OrderType.MARKET:

                                    #           Stop Limit Order            #
                                    # if close_side == OrderSide.SELL:
                                    #     stop_price = calc_with_precision(realtime_price + 5 * 10 ** -price_precision,
                                    #                                      price_precision)
                                    # else:
                                    #     stop_price = calc_with_precision(realtime_price - 5 * 10 ** -price_precision,
                                    #                                      price_precision)

                                    # request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                    #                           side=close_side,
                                    #                           ordertype=order.sl_type, stopPrice=str(stop_price),
                                    #                           quantity=str(quantity), price=str(realtime_price),
                                    #                           reduceOnly=True)

                                    if pd.isna(sl_level):
                                        exit_price = realtime_price
                                    else:
                                        exit_price = sl_level

                                    #           Limit Order             #
                                    request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                                              side=close_side,
                                                              ordertype=order.sl_type,
                                                              quantity=str(quantity), price=str(exit_price),
                                                              reduceOnly=True)

                                else:
                                    #           Market Order            #
                                    result = request_client.post_order(symbol=self.symbol, side=close_side,
                                                                       ordertype=OrderType.MARKET,
                                                                       quantity=str(quantity), reduceOnly=True)

                            except Exception as e:
                                print('error in Close Order :', e)

                                #       Check error Message     #
                                if 'Quantity less than zero' in str(e):
                                    break

                                order.sl_type = OrderType.MARKET
                                continue
                            else:
                                print('Close order listed.')

                            #       Enough time for quantity to be consumed      #
                            if not order.sl_type == OrderType.MARKET:
                                time.sleep(order.exit_execution_wait - datetime.now().second)
                            else:
                                time.sleep(1)

                            #               Check Remaining Quantity             #
                            try:
                                quantity = get_remaining_quantity(self.symbol)
                            except Exception as e:
                                print('error in get_remaining_quantity :', e)
                                continue

                            if quantity == 0.0:
                                print('Close order executed.')
                                break

                            else:
                                #           Re-Close            #
                                print('Re-Close')
                                order.sl_type = OrderType.MARKET
                                continue

                        break  # <--- Break for All Close Loop, Break Partial Limit Loop

                #       Check back-tested_Profit     #
                while 1:
                    if datetime.now().timestamp() > datetime.timestamp(stacked_df.index[-1]) + \
                            fundamental.close_complete_term:  # <-- close 데이터가 완성되기 위해 충분한 시간
                        try:
                            back_df, _ = concat_candlestick(self.symbol, self.interval, days=1, limit=order.use_rows,
                                                            timesleep=0.2, show_process=False)

                            #       realtime candlestick confirmation       #
                            if datetime.timestamp(back_df.index[-1]) > datetime.now().timestamp():
                                break

                        except Exception as e:
                            print('error in back-test_Profit :', e)

                if open_side == OrderSide.BUY:
                    calc_tmp_profit = back_df['close'].iloc[-2] / ep - self.trading_fee
                else:
                    calc_tmp_profit = ep / back_df['close'].iloc[-2] - self.trading_fee

                back_tmp_profit = 1.0

                if open_side == OrderSide.BUY:
                    if back_df['low'].iloc[-2] < df['long_ep'].iloc[self.last_index]:
                        back_tmp_profit = back_df['close'].iloc[-2] / df['long_ep'].iloc[self.last_index] - self.trading_fee
                else:
                    if back_df['high'].iloc[-2] > df['short_ep'].iloc[self.last_index]:
                        back_tmp_profit = df['short_ep'].iloc[self.last_index] / back_df['close'].iloc[-2] - self.trading_fee

                self.accumulated_back_profit *= 1 + (back_tmp_profit - 1) * leverage
                print('temporary_back_Profit : %.3f %%' % ((back_tmp_profit - 1) * leverage * 100))
                print('accumulated_back_Profit : %.3f %%' % ((self.accumulated_back_profit - 1) * 100))

                end_timestamp = int(time.time() * 1000)
                #           Get Total Income from this Trade       #
                try:
                    income = total_income(self.symbol, start_timestamp, end_timestamp)
                    self.accumulated_income += income
                except Exception as e:
                    print('error in total_income :', e)

                tmp_profit = income / available_balance
                self.accumulated_profit *= (1 + tmp_profit)
                self.calc_accumulated_profit *= 1 + (calc_tmp_profit - 1) * leverage
                print('temporary_Profit : %.3f (%.3f) %%' % (tmp_profit * 100, (calc_tmp_profit - 1) * leverage * 100))
                print('accumulated_Profit : %.3f (%.3f) %%' % ((self.accumulated_profit - 1) * 100, (self.calc_accumulated_profit - 1) * 100))
                print('accumulated_Income :', self.accumulated_income, 'USDT')


