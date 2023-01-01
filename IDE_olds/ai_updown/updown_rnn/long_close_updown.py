import os
switch_path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend"
os.chdir(switch_path)

from binance_futures_modules import *
from funcs.funcs_trader import *
from binance_futures_concat_candlestick import concat_candlestick
from easydict import EasyDict
from updown_rnn.utils import arima_profit, calc_train_days, ep_stacking
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
# from sklearn.preprocessing import MinMaxScaler
from updown_rnn.trainer import *

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.keras.backend.set_session(tf.Session(config=tf_config))

#           GPU Set         #
tf.device('/device:XLA_GPU:0')
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)


class ARIMA_Bot:

    def __init__(self, symbol, interval, threshold, tp, leverage_, initial_asset, stacked_df_on=False):

        self.last_index = -1
        self.symbol = symbol
        self.interval = interval
        self.threshold = threshold
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

        #         0. model params         #
        model_abs_path = os.path.abspath("test_set/model/classifier_45_ma7_pr3_03766.h5")
        period = 45
        t_result_path = 'npy/' + '%s_rnn_close_updown_t_result_%s.npy' % (period, self.symbol)
        pr_list_path = 'npy/' + '%s_rnn_close_updown_pr_list_%s.npy' % (period, self.symbol)
        model_path = r'%s' % model_abs_path

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
                continue

            #       Init Order side      #
            open_side = None
            close_side = None

            #       Get startTime for Total Income calc    #
            start_timestamp = int(time.time() * 1000)
            ai_survey = True
            arima_on = False
            get_first_df = False
            while 1:

                #       arima phase 도 ai survey pass 시 진행해야한다           #
                if ai_survey:  # 해당 interval 내에 arima survey 수행을 한번만 하기 위한 조건 phase
                    # try:
                        temp_time = time.time()

                        df_path = './database/%s/%s_ma7_close_entry_ai_plus.xlsx' % (self.interval, self.symbol)
                        if self.stacked_df_on:

                            try:
                                back_df
                            except Exception as e:
                                get_first_df = True
                            else:
                                first_df = back_df

                            if get_first_df:
                                first_df, _ = concat_candlestick(self.symbol, self.interval, days=1)
                                get_first_df = False

                            #       realtime candlestick confirmation       #
                            if datetime.timestamp(first_df.index[-1]) < datetime.now().timestamp():
                                get_first_df = True
                                continue

                            stacked_df = pd.read_excel(df_path, index_col=0)

                            #    use rows   #
                            #       add ep columns for append      #
                            # first_df['ep'] = np.nan
                            prev_complete_df = stacked_df.iloc[:-1, :]
                            stacked_df = prev_complete_df.append(first_df)

                            stacked_df = stacked_df[~stacked_df.index.duplicated(keep='first')].iloc[-(order.use_rows + 1 + 1):, :]

                            #         ep stacking + 1       #
                            # ep_list = ep_stacking(stacked_df.iloc[:-1, :], tp=self.tp, test_size=1, use_rows=order.use_rows)
                            # #       add upper ep_list to stacked_df     #
                            # #       time index 와 ep 값을 매칭시켜, 누락에 대응한다       #
                            # #       누락 발생시 period size 로 ep stacking 진행         #
                            # stacked_df['ep'].iloc[-len(ep_list) - 1: -1] = ep_list
                            #
                            # #   temp_ep 에 nan value 를 포함시키지 않기 위한 검수    #)
                            # if np.isnan(np.sum(stacked_df['ep'].iloc[-period - 1:-1].values)):
                            #     self.stacked_df_on = False

                            # print('stacked_df.tail(50) :\n', stacked_df.tail(50))
                            # quit()

                        if not self.stacked_df_on:

                            days = calc_train_days(interval=self.interval, use_rows=(order.use_rows + period + 1))
                            stacked_df, _ = concat_candlestick(self.symbol, self.interval, days=days, timesleep=0.2)
                            stacked_df = stacked_df.iloc[-(order.use_rows + period + 1):, :]
                            self.stacked_df_on = True

                            #         ep stacking + period      #
                            # stacked_df['ep'] = np.nan
                            # ep_list = ep_stacking(stacked_df.iloc[:-1, :], tp=self.tp, test_size=period, use_rows=order.use_rows)
                            # stacked_df['ep'].iloc[-len(ep_list) - 1:-1] = ep_list

                            # print('stacked_df.tail(50) :\n', stacked_df.tail(50)

                        #         AI survey phase       #
                        test_result = retrain_and_pred(stacked_df, symbol=self.symbol)
                        ai_survey = False

                        #       rewrite last timestamp      #
                        with open('updown_rnn/survey_logger/%s.txt' % self.symbol, 'w') as log_file:
                            log_file.write(str(datetime.now().timestamp()))

                        best_thr = get_best_thr(t_result_path, pr_list_path)

                        if test_result[:, [1]] > best_thr:
                            # 거래 진행, 아니면 거래 대기

                            #         ai phase 의 종속 조건 수행절      #
                            #         ARIMA survey phase          #
                            df, pred_close, err_range = arima_profit(stacked_df.iloc[:-1, :], tp=self.tp,
                                                                     leverage=self.leverage_)  # <-- we should use complete data
                            print('stacked_df.index[-1] :', stacked_df.index[-1],
                                  'checking time : %.2f' % (time.time() - temp_time))
                            print('pred_close :', pred_close)
                            arima_on = True

                    # except Exception as e:
                    #     print('Error in arima_profit :', e)
                    #     continue

                if arima_on:

                    #           Continuously check open condition          #
                    #      Get realtime market price & compare with trigger price      #
                    # try:
                    #     realtime_price = get_market_price_v2(self.sub_client)
                    #
                    # except Exception as e:
                    #     print('Error in get_market_price :', e)
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
                    ai_survey = True
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
                    print('Error in get_availableBalance :', e)
                    print()
                    continue

                #          Get available quantity         #
                quantity = available_balance / ep * leverage
                quantity = calc_with_precision(quantity, quantity_precision)
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
                            print('Error in Open Order :', e)

                            open_retry_cnt += 1
                            if 'insufficient' in str(e):

                                try:
                                    max_available_balance = get_availableBalance()
                                    #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                                    self.over_balance = available_balance
                                    available_balance = max_available_balance * 0.9
                                    #          Get available quantity         #
                                    quantity = available_balance / ep * leverage
                                    quantity = calc_with_precision(quantity, quantity_precision)
                                    print('available_balance (temp) :', available_balance)
                                    print('quantity :', quantity)

                                except Exception as e:
                                    print('Error in get_availableBalance() :', e)

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

                    while 1:    # <-- loop for complete open order
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

                            open_retry_cnt += 1
                            if 'insufficient' in str(e):

                                try:
                                    max_available_balance = get_availableBalance()
                                    #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                                    self.over_balance = available_balance
                                    available_balance = max_available_balance * 0.9
                                    #          Get available quantity         #
                                    quantity = available_balance / ep * leverage
                                    quantity = calc_with_precision(quantity, quantity_precision)
                                    print('available_balance (temp) :', available_balance)
                                    print('quantity :', quantity)

                                except Exception as e:
                                    print('Error in get_availableBalance() :', e)

                            elif open_retry_cnt > 100:
                                print('open_retry_cnt over 100')
                                quit()
                            continue
                        else:
                            print('Open order listed.')
                            break

                    #       Set Order Execute Time & Check breakout_qty_ratio         #
                    while 1:

                        #       1분전까지 체결 대기     #
                        if datetime.now().timestamp() + order.entry_execution_wait > datetime.timestamp(stacked_df.index[-1]):
                            break

                        #           Check order type change condition           #
                        #      Get realtime market price & compare with trigger price      #
                        # try:
                        #     realtime_price = get_market_price_v2(self.sub_client)
                        # except Exception as e:
                        #     print('Error in get_market_price :', e)
                        #     continue

                        #          Check order type change condition       #
                        # if open_side == OrderSide.SELL:
                        #     if realtime_price < pred_close - err_range:
                        #
                        #         #       체결 내역이 존재하지 않으면,        # <-- api limit 을 고려해 내부로 가져옴
                        #         try:
                        #             exec_quantity = get_remaining_quantity(self.symbol)
                        #         except Exception as e:
                        #             print('Error in exec_quantity check :', e)
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
                        #             print('Error in exec_quantity check :', e)
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
                #     print('Error in remaining_order_check :', e)

                # if remained_orderId is not None:
                #           regardless to position exist, cancel all open orders       #
                try:
                    result = request_client.cancel_all_orders(symbol=self.symbol)
                except Exception as e:
                    print('Error in cancel remaining open order :', e)

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
                    print('Error in exec_quantity check :', e)
                    continue
                break

            if exec_quantity == 0.0:

                while 1:
                    #        체결량 없을시 연속적인 open order 를 방지해야한다        #
                    if datetime.now().timestamp() > datetime.timestamp(stacked_df.index[-1]) + fundamental.close_complete_term:  # close 데이터가 완성되기 위해 충분한 시간

                        #       Check back-tested_Profit     #
                        while 1:
                            try:
                                back_df, _ = concat_candlestick(self.symbol, self.interval, days=1)

                                #       realtime candlestick confirmation       #
                                if datetime.timestamp(back_df.index[-1]) > datetime.now().timestamp():
                                    break

                            except Exception as e:
                                print('Error in back-test_Profit :', e)

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
                                        print('Error in cancel remaining TP Close order :', e)
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
                    if datetime.now().timestamp() > datetime.timestamp(stacked_df.index[-1]) + \
                            fundamental.close_complete_term:  # <-- close 데이터가 완성되기 위해 충분한 시간
                        try:
                            back_df, _ = concat_candlestick(self.symbol, self.interval, days=1)

                            #       realtime candlestick confirmation       #
                            if datetime.timestamp(back_df.index[-1]) > datetime.now().timestamp():
                                break

                        except Exception as e:
                            print('Error in back-test_Profit :', e)

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

                # #           save realtime trade log         #
                # try:
                #     total_result = np.load(t_result_path)
                #     total_result = list(total_result)
                #     pr_list = np.load(pr_list_path)
                #     pr_list = list(pr_list)
                #     print("----------------------- total_result & pr_list loaded ! -----------------------")
                #     print("len(total_result) :", len(total_result))
                #     print("len(pr_list) :", len(pr_list))
                #
                #     total_result.append(test_result)
                #     float_pr = (back_tmp_profit - 1) * leverage + 1
                #     pr_list.append(float_pr)
                #
                #     np.save(t_result_path, np.array(total_result[-order.use_rows:]))
                #     np.save(pr_list_path, np.array(pr_list[-order.use_rows:]))
                #     print("----------------------- total_result & pr_list saved ! -----------------------")
                #
                # except Exception as e:
                #     print('Error in load total_result & pr_list :', e)

                end_timestamp = int(time.time() * 1000)
                #           Get Total Income from this Trade       #
                try:
                    income = total_income(self.symbol, start_timestamp, end_timestamp)
                    self.accumulated_income += income
                except Exception as e:
                    print('Error in total_income :', e)

                tmp_profit = income / available_balance
                self.accumulated_profit *= (1 + tmp_profit)
                self.calc_accumulated_profit *= 1 + (calc_tmp_profit - 1) * leverage
                print('temporary_Profit : %.3f (%.3f) %%' % (tmp_profit * 100, (calc_tmp_profit - 1) * leverage * 100))
                print('accumulated_Profit : %.3f (%.3f) %%' % ((self.accumulated_profit - 1) * 100, (self.calc_accumulated_profit - 1) * 100))
                print('accumulated_Income :', self.accumulated_income, 'USDT')
                print()

            #           save realtime trade log         #
            try:
                total_result = np.load(t_result_path)
                total_result = list(total_result)
                pr_list = np.load(pr_list_path)
                pr_list = list(pr_list)
                print("----------------------- total_result & pr_list loaded ! -----------------------")
                print("len(total_result) :", len(total_result))
                print("len(pr_list) :", len(pr_list))

                total_result.append(test_result)
                float_pr = (back_tmp_profit - 1) * leverage + 1
                pr_list.append(float_pr)

                np.save(t_result_path, np.array(total_result[-order.use_rows:]))
                np.save(pr_list_path, np.array(pr_list[-order.use_rows:]))
                print("----------------------- total_result & pr_list saved ! -----------------------")

            except Exception as e:
                print('Error in load total_result & pr_list :', e)

