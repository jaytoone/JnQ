from binance_f import RequestClient
from binance_f.constant.test import *
from binance_f.base.printobject import *
from binance_f.model.constant import *
from Binance_Futures_Modules import *
import time
from datetime import datetime
from Funcs_For_Trade import *
from Funcs_Indicator import *

request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)

symbol = "COMPUSDT"
bar_close_second = 59   # <-- in 1m bar
accumulated_income = 0.0

#            Initialize Part by Symbol           #
#         1. Leverage type => Isolated          #
try:
    request_client.change_margin_type(symbol=symbol, marginType=FuturesMarginType.ISOLATED)
except Exception as e:
    print('Error in change_margin_type :', e)
else:
    print('Leverage type --> Isolated')

#         2. Confirm Limit Leverage          #
try:
    limit_leverage = get_limit_leverage(symbol_=symbol)
except Exception as e:
    print('Error in get_limit_leverage :', e)
else:
    print('Limit Leverage :', limit_leverage)

#         3. Get price, volume precision       #
try:
    price_precision, quantity_precision = get_precision(symbol)
except Exception as e:
    print('Error in get price & volume precision :', e)
else:
    print('price_precision :', price_precision)
    print('quantity_precision :', quantity_precision)

print()
print('Trade Running Now...')
while 1:

    #       Init Order side      #
    open_side = None
    close_side = None

    #       Get startTime for Total Income calc    #
    start_timestamp = int(time.time() * 1000)

    while 1:

        #           Wait for bar closing time         #
        if datetime.now().second >= bar_close_second:
            pass
        else:
            continue

        #           Open Condition          #
        try:
            df = request_client.get_candlestick_data(symbol=symbol, interval=CandlestickInterval.MIN1,
                                                     startTime=None, endTime=None, limit=1500)
        except Exception as e:
            print('Error in get_candlestick_data :', e)
            continue

        period1, period2, period3 = 30, 60, 120

        df['Fisher1'] = fisher(df, period=period1)
        df['Fisher2'] = fisher(df, period=period2)
        df['Fisher3'] = fisher(df, period=period3)

        tc_upper = 0.35
        tc_lower = -tc_upper

        df['Fisher1_Trend'] = fisher_trend(df, 'Fisher1', 0, 0)
        df['Fisher2_Trend'] = fisher_trend(df, 'Fisher2', tc_upper, tc_lower)
        df['Fisher3_Trend'] = fisher_trend(df, 'Fisher3', tc_upper, tc_lower)

        upper = 0.5
        lower = -upper

        i = -1

        #               Long                #
        if df['Fisher2_Trend'].iloc[i] == 'Long' and df['Fisher3_Trend'].iloc[i] == 'Long':
            if df['Fisher1'].iloc[i - 2] > df['Fisher1'].iloc[i - 1] < df['Fisher1'].iloc[i]:
                if df['Fisher1'].iloc[i - 1] < lower:
                    open_side = OrderSide.SELL

        #               Short               #
        if df['Fisher2_Trend'].iloc[i] == 'Short' and df['Fisher3_Trend'].iloc[i] == 'Short':
            if df['Fisher1'].iloc[i - 2] < df['Fisher1'].iloc[i - 1] > df['Fisher1'].iloc[i]:
                if df['Fisher1'].iloc[i - 1] > upper:
                    open_side = OrderSide.BUY

        if open_side is not None:
            break
        # ---- until open condition satisfied... ----

    print('We Got Open Signal.')
    print('open_side :', open_side)

    #             Set Leverage              #
    sl_level = None
    offset_ratio = 1 / 2
    target_percentage = 0.05
    backing_i = i
    while 1:
        if df['Fisher1_Trend'].iloc[backing_i] != df['Fisher1_Trend'].iloc[backing_i - 1]:

            low = min(df['low'].iloc[backing_i:])
            high = max(df['high'].iloc[backing_i:])

            if open_side == OrderSide.BUY:
                sl_level = low - (high - low) * offset_ratio
            else:
                sl_level = high + (high - low) * offset_ratio
            break

        backing_i -= 1

    try:
        ep = get_market_price(symbol)
    except Exception as e:
        print('Error in get_market_price :', e)

    sl_level = calc_with_precision(sl_level, price_precision)
    leverage = int(target_percentage / (abs(ep - sl_level) / ep))
    leverage = min(limit_leverage, leverage)
    print('sl_level :', sl_level)
    print('ep :', ep)
    print('leverage :', leverage)

    try:
        request_client.change_initial_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        print('Error in change_initial_leverage :', e)
    else:
        print('leverage changed -->', leverage)

    #          Get availableBalance          #
    try:
        available_balance = get_availableBalance() * 0.9
    except Exception as e:
        print('Error in get_availableBalance :', e)

    #          Define Start Asset          #
    if accumulated_income == 0.0:
        start_asset = available_balance

    #          Get available quantity         #
    quantity = available_balance / ep * leverage
    str_quantity = str(calc_with_precision(quantity, quantity_precision))
    print('available_balance :', available_balance)
    print('str_quantity :', str_quantity)

    #           Open            #
    try:
        #           Limit Order             #
        # result = request_client.post_order(timeInForce=TimeInForce.GTC, symbol="SXPUSDT", side=open_side, ordertype=OrderType.LIMIT,
        #                                    quantity=str_quantity, price=0.9000)
        #           Market Order            #
        request_client.post_order(symbol=symbol, side=open_side, ordertype=OrderType.MARKET, quantity=str_quantity)
    except Exception as e:
        print('Error in Open Order :', e)
    else:
        print('Open order succeed.')

    #       Enough time for quantity Consuming      #
    time.sleep(60 - bar_close_second)

    #           Remaining Order check            #
    remained_orderId = None

    try:
        remained_orderId = remaining_order_check()
    except Exception as e:
        print('Error in remaining_order_check :', e)

    if remained_orderId is not None:
        #           If Remained position exist, Cancel it       #
        try:
            result = request_client.cancel_order(symbol=symbol, orderId=remained_orderId)
        except Exception as e:
            print('Error in Cancel remained open order :', e)

    #       position 이 존재하면 진행하는 걸로      #
    exec_quantity = 0.0

    try:
        exec_quantity = get_remaining_quantity(symbol)
    except Exception as e:
        print('Error in exec_quantity check :', e)

    if exec_quantity != 0.0:

        print('Open order executed.')
        print()

        #           side Change         #
        if open_side == OrderSide.BUY:
            close_side = OrderSide.SELL
        else:
            close_side = OrderSide.BUY

        close_count = 2
        tp_count = 0
        sl_fisher_level = 0
        sl_level_ = sl_level

        #           Prevent close at open bar           #
        time.sleep(70)
        m = -1
        while 1:  # <--- Loop for 2 TP

            TP_switch, SL_switch = False, False
            while 1:

                #           Wait for bar closing time         #
                if datetime.now().second >= bar_close_second:
                    pass
                else:
                    continue

                #           Close Condition          #
                try:
                    df = request_client.get_candlestick_data(symbol=symbol, interval=CandlestickInterval.MIN1,
                                                             startTime=None, endTime=None, limit=1500)
                except Exception as e:
                    print('Error in get_candlestick_data :', e)
                    continue

                df['Fisher1'] = fisher(df, period=period1)
                df['Fisher2'] = fisher(df, period=period2)
                df['Fisher3'] = fisher(df, period=period3)

                df['Fisher1_Trend'] = fisher_trend(df, 'Fisher1', 0, 0)
                df['Fisher2_Trend'] = fisher_trend(df, 'Fisher2', tc_upper, tc_lower)
                df['Fisher3_Trend'] = fisher_trend(df, 'Fisher3', tc_upper, tc_lower)

                #               Long                #
                if open_side == OrderSide.SELL:
                    #           TP by Signal            #
                    if df['Fisher1'].iloc[m - 2] < df['Fisher1'].iloc[m - 1] > df['Fisher1'].iloc[m]:
                        if df['Fisher1'].iloc[m - 1] > upper:
                            TP_switch = True

                    #            SL by Price           #
                    if df['low'].iloc[m] < sl_level_:
                        SL_switch = True

                    #           SL by Signal            #
                    if close_count == 2:
                        # if df['Fisher1'].iloc[m - 1] > -sl_fisher_level_none_tp > df['Fisher1'].iloc[m]:
                        #     trade_state_close.iloc[m] = 'Signal SL'
                        #     SL_switch = True
                        if df['Fisher2_Trend'].iloc[m] == 'Short':
                            SL_switch = True

                    if close_count == 1:
                        if df['Fisher1'].iloc[m - 1] > -sl_fisher_level > df['Fisher1'].iloc[m]:
                            SL_switch = True

                #               Short               #
                else:
                    #           TP by Signal            #
                    if df['Fisher1'].iloc[m - 2] > df['Fisher1'].iloc[m - 1] < df['Fisher1'].iloc[m]:
                        if df['Fisher1'].iloc[m - 1] < lower:
                            TP_switch = True

                    #           SL by Price           #
                    if df['high'].iloc[m] > sl_level_:
                        SL_switch = True

                    #           SL by Signal            #
                    if close_count == 2:
                        # if df['Fisher1'].iloc[m - 1] < sl_fisher_level_none_tp < df['Fisher1'].iloc[m]:
                        #     trade_state_close.iloc[m] = 'Signal SL'
                        #     SL_switch = True
                        if df['Fisher2_Trend'].iloc[m] == 'Long':
                            SL_switch = True

                    if close_count == 1:
                        if df['Fisher1'].iloc[m - 1] < sl_fisher_level < df['Fisher1'].iloc[m]:
                            SL_switch = True

                if TP_switch or SL_switch:
                    break
                # --- wait until close condition satisfied... ---

            print('We Got Close Signal.')

            if TP_switch:
                close_count -= 1
                tp_count += 1
            else:
                close_count = 0

            print('close count(remain) :', close_count)

            while 1:  # <--- This Loop for Re-Close

                #          Get Remaining quantity         #
                #           1. tp_count == 1, quantity = 1 / 2 * quantity        #
                #           2. tp_count == 0, close all remaining quantity           #
                try:
                    quantity = get_remaining_quantity(symbol)
                except Exception as e:
                    print('Error in get_remaining_quantity :', e)
                    continue

                if tp_count == 1:
                    quantity *= 1 / 2

                str_quantity = str(calc_with_precision(quantity, quantity_precision))
                print('str_quantity :', str_quantity)

                #           Close             #
                #           1. Order side should be Opposite side to the Open          #
                #           2. ReduceOnly = 'true'       #
                try:
                    #           Limit Order             #
                    # result = request_client.post_order(timeInForce=TimeInForce.GTC, symbol="SXPUSDT", side=close_side, ordertype=OrderType.LIMIT,
                    #                                    quantity=str_quantity, price=0.9000)
                    #           Market Order            #
                    str_quantity = "{:0.0{}f}".format(quantity, quantity_precision)
                    request_client.post_order(symbol=symbol, side=close_side, ordertype=OrderType.MARKET,
                                              quantity=str_quantity, reduceOnly='true')
                except Exception as e:
                    print('Error in Close Order :', e)
                    continue
                else:
                    print('Close order succeed.')

                #       Enough time for quantity Consuming      #
                time.sleep(60 - bar_close_second)

                #           Remaining Order check            #
                try:
                    remained_orderId = remaining_order_check()
                except Exception as e:
                    print('Error in remaining_order_check :', e)

                if remained_orderId is not None:
                    #           Re-Close            #
                    print('Re-Close')
                    continue
                else:
                    print('Close order executed.')
                    break

            if close_count == 0:

                end_timestamp = int(time.time() * 1000)
                #           Get Total Income from this Trade       #
                try:
                    accumulated_income += total_income(symbol, start_timestamp, end_timestamp)
                except Exception as e:
                    print('Error in total_income :', e)
                print('accumulated_income :', accumulated_income, 'USDT')
                print('accumulated_profit : %.2f %%' % (accumulated_income / start_asset * 100))
                print()
                break   # <--- Break for All Close Loop
