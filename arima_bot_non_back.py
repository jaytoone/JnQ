from binance_futures_modules import *
from funcs.funcs_for_trade import *
from binance_futures_concat_candlestick import concat_candlestick
from easydict import EasyDict
from binance_futures_arima_modules import arima_profit, calc_train_days
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)


class ARIMA_Bot:

    def __init__(self, symbol, interval, tp, leverage_, initial_asset, stacked_df_on=False):

        self.last_index = -1
        self.symbol = symbol
        self.interval = interval
        self.tp, self.leverage_ = tp, leverage_
        self.initial_asset = initial_asset
        self.stacked_df_on = stacked_df_on

        self.trading_fee = 0.0006
        self.accumulated_income = 0.0
        self.accumulated_profit = 1.0
        self.accumulated_back_profit = 1.0

        sub_client.subscribe_aggregate_trade_event(symbol.lower(), callback, error)
        self.sub_client = sub_client

    def run(self):

        #         1. Leverage type => Isolated          #
        try:
            request_client.change_margin_type(symbol=self.symbol, marginType=FuturesMarginType.ISOLATED)
        except Exception as e:
            print('Error in change_margin_type :', e)
        else:
            print('Leverage type --> Isolated')

        #         2. Confirm Limit Leverage          #
        try:
            limit_leverage = get_limit_leverage(symbol_=self.symbol)
        except Exception as e:
            print('Error in get_limit_leverage :', e)
            quit()
        else:
            print('Limit Leverage :', limit_leverage)

        print()
        print('Trade Running Now...')

        while 1:

            #       Configuration       #
            with open('binance_futures_bot_config.json', 'r') as cfg:
                config = EasyDict(json.load(cfg))

            fundamental = config.FUNDMTL
            order = config.ORDER

            #       Run Decision        #
            if not fundamental.run:
                break

            #       Init Order side      #
            open_side = None
            close_side = None

            #       Get startTime for Total Income calc    #
            start_timestamp = int(time.time() * 1000)
            init_arima = True

            while 1:

                if init_arima:
                    try:
                        temp_time = time.time()

                        df_path = './candlestick_concated/%s/%s.xlsx' % (self.interval, self.symbol)
                        if self.stacked_df_on:

                            first_df, _ = concat_candlestick(self.symbol, self.interval, days=1)
                            stacked_df = pd.read_excel(df_path, index_col=0)

                            #    use rows   #
                            stacked_df = stacked_df.append(first_df)
                            stacked_df = stacked_df[~stacked_df.index.duplicated(keep='first')].iloc[-order.use_rows:, :]

                        else:
                            days = calc_train_days(interval=self.interval, use_rows=order.use_rows)
                            stacked_df, _ = concat_candlestick(self.symbol, self.interval, days=days, timesleep=0.2)
                            stacked_df = stacked_df.iloc[-order.use_rows:, :]
                            self.stacked_df_on = True

                        stacked_df.to_excel(df_path)  # <-- save new_stacked_df
                        # quit()

                        #         ARIMA process          #
                        df, pred_close, err_range = arima_profit(stacked_df.iloc[:-1, :], tp=self.tp, leverage=self.leverage_)  # <-- we should use complete data
                        init_arima = False
                        print('stacked_df.index[-1] :', stacked_df.index[-1], 'checking time : %.2f' % (time.time() - temp_time))
                        print('pred_close, err_range :', pred_close, err_range)

                    except Exception as e:
                        print('Error in arima_profit :', e)
                        continue

                #           Continuously check open condition          #
                #      Get realtime market price & compare with trigger price      #
                try:
                    realtime_price = get_market_price_v2(self.sub_client)

                except Exception as e:
                    print('Error in get_market_price :', e)
                    continue

                if realtime_price < pred_close - err_range:
                    open_side = OrderSide.BUY

                elif realtime_price > pred_close + err_range:
                    open_side = OrderSide.SELL

                if open_side is not None:
                    break

                time.sleep(fundamental.realtime_term)  # <-- term for realtime market function

                #       check tick change      #
                #       stacked_df의 마지막 timestamp index 보다 current timestamp 이 커지면 init arima       #
                if datetime.timestamp(stacked_df.index[-1]) < datetime.now().timestamp():
                    init_arima = True
                    print('stacked_df[-1] timestamp :', datetime.timestamp(stacked_df.index[-1]))  # <-- code proof
                    print('current timestamp :', datetime.now().timestamp())

                    time.sleep(fundamental.close_complete_term)   # <-- term for close completion

            print('realtime_price :', realtime_price)
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
                        print('Error in get_market_price :', e)
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
                    print('Error in get price & volume precision :', e)
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
                    print('Error in change_initial_leverage :', e)
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

                            if available_balance > max_available_balance:
                                available_balance = max_available_balance * 0.95

                        except Exception as e:
                            print('Error in get_availableBalance :', e)

                #          Get available quantity         #
                quantity = available_balance / ep * leverage
                quantity = calc_with_precision(quantity, quantity_precision)
                print('available_balance :', available_balance)
                print('quantity :', quantity)

                #           Open            #
                orderside_changed = False

                if order.entry_type == OrderType.MARKET:
                    try:
                        #           Market Order            #
                        request_client.post_order(symbol=self.symbol, side=open_side, ordertype=OrderType.MARKET,
                                                  quantity=str(quantity))
                    except Exception as e:
                        print('Error in Open Order :', e)

                        if 'insufficient' in e:
                            quit()
                        continue
                    else:
                        print('Open order listed.')

                    #       Enough time for quantity Consuming      #
                    time.sleep(60 - fundamental.bar_close_second)

                else:
                    #       If Limit Order used, Set Order Execute Time & Check Remaining Order         #
                    try:
                        #           Limit Order            #
                        result = request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.symbol,
                                                           side=open_side,
                                                           ordertype=order.entry_type,
                                                           quantity=str(quantity), price=str(ep),
                                                           reduceOnly=False)
                    except Exception as e:
                        print('Error in Open Order :', e)

                        if 'insufficient' in e:
                            quit()
                        continue
                    else:
                        print('Open order listed.')

                    #       Set Order Execute Time & Check breakout_qty_ratio         #
                    #       Check Realtime price cross sl_level         #
                    exec_start = time.time()
                    while 1:

                        #       1분전까지 체결 대기     #
                        if datetime.now().timestamp() + order.entry_execution_wait > datetime.timestamp(stacked_df.index[-1]):
                            break

                        #           Check order type change condition           #
                        #      Get realtime market price & compare with trigger price      #
                        try:
                            realtime_price = get_market_price_v2(self.sub_client)
                        except Exception as e:
                            print('Error in get_market_price :', e)
                            continue

                        try:
                            exec_quantity = get_remaining_quantity(self.symbol)
                        except Exception as e:
                            print('Error in exec_quantity check :', e)
                            continue

                        #       체결 내역이 존재하지 않으면,        #
                        if exec_quantity == 0.0:

                            #          Check order type change condition       #
                            if open_side == OrderSide.SELL:
                                if realtime_price < pred_close - err_range:
                                    open_side = OrderSide.BUY
                                    orderside_changed = True
                                    break

                            else:
                                if realtime_price > pred_close + err_range:
                                    open_side = OrderSide.SELL
                                    orderside_changed = True
                                    break

                        time.sleep(fundamental.realtime_term)  # <-- for realtime price function

                #           Remaining Order check            #
                remained_orderId = None

                try:
                    remained_orderId = remaining_order_check(self.symbol)
                except Exception as e:
                    print('Error in remaining_order_check :', e)

                if remained_orderId is not None:
                    #           If Remained position exist, Cancel it       #
                    try:
                        # result = request_client.cancel_order(symbol=self.symbol, orderId=remained_orderId)
                        result = request_client.cancel_all_orders(symbol=self.symbol)
                    except Exception as e:
                        print('Error in Cancel remained open order :', e)

                if orderside_changed:
                    first_iter = False
                    print('orderside_changed :', orderside_changed)
                    print()
                    continue
                else:
                    break

            if exec_quantity == 0.0:
                while 1:
                    #        체결량 없을시 연속적인 open order 를 방지해야한다        #
                    if datetime.now().timestamp() > datetime.timestamp(stacked_df.index[-1]) + fundamental.close_complete_term:  # close 데이터가 완성되기 위해 충분한 시간
                        break

                income = 0
                print()
                continue

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
                            print('Error in partial_limit :', e)
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
                                print('Error in get_market_price :', e)
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
                                    print('Error in remaining_order_check :', e)
                                    continue

                                if remained_orderId is not None:
                                    #           If Remained position exist, Cancel it       #
                                    try:
                                        result = request_client.cancel_all_orders(symbol=self.symbol)
                                    except Exception as e:
                                        print('Error in Cancel remained TP Close order :', e)
                                        continue

                                cancel_tp = True

                            #          Get Remaining quantity         #
                            try:
                                quantity = get_remaining_quantity(self.symbol)
                            except Exception as e:
                                print('Error in get_remaining_quantity :', e)
                                continue

                            #           Get price, volume precision       #
                            try:
                                _, quantity_precision = get_precision(self.symbol)
                            except Exception as e:
                                print('Error in get price & volume precision :', e)
                            else:
                                print('quantity_precision :', quantity_precision)

                            quantity = calc_with_precision(quantity, quantity_precision)
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
                                print('Error in Close Order :', e)

                                #       Check Error Message     #
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
                                print('Error in get_remaining_quantity :', e)
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
                    if datetime.now().second > fundamental.close_complete_term + 5:  # <-- close 데이터가 완성되기 위해 충분한 시간
                        try:
                            back_df, _ = concat_candlestick(self.symbol, self.interval, days=1)

                        except Exception as e:
                            print('Error in back-test_Profit :', e)

                        else:
                            break

                if open_side == OrderSide.BUY:
                    back_tmp_profit = back_df['close'].iloc[-2] / ep - self.trading_fee
                else:
                    back_tmp_profit = ep / back_df['close'].iloc[-2] - self.trading_fee

                self.accumulated_back_profit *= 1 + (back_tmp_profit - 1) * leverage
                print('temporary_back_Profit : %.3f %%' % ((back_tmp_profit - 1) * leverage * 100))
                print('accumulated_back_Profit : %.3f %%' % ((self.accumulated_back_profit - 1) * 100))

                end_timestamp = int(time.time() * 1000)
                #           Get Total Income from this Trade       #
                try:
                    income = total_income(self.symbol, start_timestamp, end_timestamp)
                    self.accumulated_income += income
                except Exception as e:
                    print('Error in total_income :', e)

                tmp_profit = income / available_balance
                self.accumulated_profit *= (1 + tmp_profit)
                print('temporary_Profit : %.3f %%' % (tmp_profit * 100))
                print('accumulated_Profit : %.3f %%' % ((self.accumulated_profit - 1) * 100))
                print('accumulated_Income :', self.accumulated_income, 'USDT')
                print()

