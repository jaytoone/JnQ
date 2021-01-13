from Binance_Futures_Modules import *
from Funcs_For_Trade import *
# from Funcs_Indicator import *
from Funcs_for_Open_Ichimoku_modified import profitage
from Funcs_for_Close_Ichimoku_modified import profitage as c_profitage
from Binance_Futures_concat_candlestick import concat_candlestick
from easydict import EasyDict


#            Initialize Part by Symbol           #
#            First Configuration       #
with open('Binance_Futures_Bot_Config.json', 'r') as cfg:
    config = EasyDict(json.load(cfg))
fundamental = config.FUNDMTL
accumulated_income = fundamental.accumulated_income

#         1. Leverage type => Isolated          #
try:
    request_client.change_margin_type(symbol=fundamental.symbol, marginType=FuturesMarginType.ISOLATED)
except Exception as e:
    print('Error in change_margin_type :', e)
else:
    print('Leverage type --> Isolated')

#         2. Confirm Limit Leverage          #
try:
    limit_leverage = get_limit_leverage(symbol_=fundamental.symbol)
except Exception as e:
    print('Error in get_limit_leverage :', e)
    quit()
else:
    print('Limit Leverage :', limit_leverage)

print()
print('Trade Running Now...')


while 1:

    #       Configuration       #
    with open('Binance_Futures_Bot_Config.json', 'r') as cfg:
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

    while 1:

        time.sleep(.5)

        #           Wait for bar closing time         #
        if datetime.now().second >= fundamental.bar_close_second:
            pass
        else:
            continue

        #       Check Condition      #
        try:
            first_df, _ = concat_candlestick(fundamental.symbol, '1m', days=1)
            second_df, _ = concat_candlestick(fundamental.symbol, '3m', days=1)
            third_df, _ = concat_candlestick(fundamental.symbol, '30m', days=1)

            df = profitage(first_df, second_df, third_df, label_type=LabelType.TEST)
            print('df.index[-1] :', df.index[-1])

        except Exception as e:
            print('Error in making Test set :', e)
            continue

        last_i = -1

        #           Open Condition          #
        #               Long & Short               #
        if df['trade_state'].iloc[last_i] == 1:
            open_side = OrderSide.BUY

        elif df['trade_state'].iloc[last_i] == 0:
            open_side = OrderSide.SELL

        if open_side is not None:
            break
        # ---- until open condition satisfied... ----

    print('We Got Open Signal.')
    print('open_side :', open_side)

    #             Set Leverage              #
    target_percentage = 0.05

    if order.entry_type == OrderType.MARKET:
        #             ep with Market            #
        try:
            ep = get_market_price(fundamental.symbol)
        except Exception as e:
            print('Error in get_market_price :', e)
    else:
        #             ep with Limit             #
        ep = df['ep'].iloc[last_i]

    #           TP & SL             #
    sl_level = df['sl_level'].iloc[last_i]
    tp_level = df['tp_level'].iloc[last_i]
    leverage = df['leverage'].iloc[last_i]

    partial_num = 3
    partial_qty_divider = 1.5

    tp_list = list()
    if open_side == OrderSide.BUY:

        for part_i in range(partial_num, 0, -1):
            tmp_tp_level = ep + abs(tp_level - ep) * (part_i / partial_num)
            tp_list.append(tmp_tp_level)

        if order.entry_type == OrderType.MARKET:
            if ep <= sl_level:
                print('ep <= level : %s <= %s' % (ep, sl_level))
                continue
    else:

        for part_i in range(partial_num, 0, -1):
            tmp_tp_level = ep - abs(tp_level - ep) * (part_i / partial_num)
            tp_list.append(tmp_tp_level)

        if order.entry_type == OrderType.MARKET:
            if ep >= sl_level:
                print('ep >= level : %s >= %s' % (ep, sl_level))
                continue

    #         Get price, volume precision --> 가격 변동으로 인한 precision 변동 가능성       #
    try:
        price_precision, quantity_precision = get_precision(fundamental.symbol)
    except Exception as e:
        print('Error in get price & volume precision :', e)
        continue
    else:
        print('price_precision :', price_precision)
        print('quantity_precision :', quantity_precision)

    sl_level = calc_with_precision(sl_level, price_precision)
    tp_list = list(map(lambda x: calc_with_precision(x, price_precision), tp_list))
    # leverage = int(target_percentage / (abs(ep - sl_level) / ep))
    leverage = min(limit_leverage, leverage)
    print('ep :', ep)
    print('sl_level :', sl_level)
    print('tp_list :', tp_list)
    print('leverage :', leverage)

    try:
        request_client.change_initial_leverage(symbol=fundamental.symbol, leverage=leverage)
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
    quantity = calc_with_precision(quantity, quantity_precision)
    print('available_balance :', available_balance)
    print('quantity :', quantity)

    #           Open            #
    if order.entry_type == OrderType.MARKET:
        try:
            #           Market Order            #
            request_client.post_order(symbol=fundamental.symbol, side=open_side, ordertype=OrderType.MARKET,
                                      quantity=str(quantity))
        except Exception as e:
            print('Error in Open Order :', e)
        else:
            print('Open order succeed.')

        #       Enough time for quantity Consuming      #
        time.sleep(60 - fundamental.bar_close_second)

    else:
        #       If Limit Order used, Set Order Execute Time & Check Remaining Order         #
        try:
            #           Limit Order            #
            result = request_client.post_order(timeInForce=TimeInForce.GTC, symbol=fundamental.symbol, side=open_side,
                                               ordertype=order.entry_type,
                                               quantity=str(quantity), price=str(ep),
                                               reduceOnly=False)
        except Exception as e:
            print('Error in Open Order :', e)
        else:
            print('Open order succeed.')

        #       Set Order Execute Time & Check breakout_qty_ratio         #
        #       Check Realtime price cross sl_level         #
        exec_start = time.time()
        while 1:
            if time.time() - exec_start > order.entry_execution_wait - datetime.now().second:
                break

            try:
                exec_quantity = get_remaining_quantity(fundamental.symbol)
            except Exception as e:
                print('Error in exec_quantity check :', e)

            if exec_quantity > quantity * order.breakout_qty_ratio:
                break

            #           realtime price compare with sl_level        #
            # try:
            #     realtime_price = get_market_price(fundamental.symbol)
            # except Exception as e:
            #     print('Error in get_market_price :', e)
            #
            # if open_side == OrderSide.BUY:
            #     if realtime_price < sl_level:
            #         break
            # else:
            #     if realtime_price > sl_level:
            #         break

            time.sleep(.5)

    #           Remaining Order check            #
    remained_orderId = None

    try:
        remained_orderId = remaining_order_check()
    except Exception as e:
        print('Error in remaining_order_check :', e)

    if remained_orderId is not None:
        #           If Remained position exist, Cancel it       #
        try:
            # result = request_client.cancel_order(symbol=fundamental.symbol, orderId=remained_orderId)
            result = request_client.cancel_all_orders(symbol=fundamental.symbol)
        except Exception as e:
            print('Error in Cancel remained open order :', e)

    #       position 이 존재하면 진행하는 걸로      #
    exec_quantity = 0.0

    try:
        exec_quantity = get_remaining_quantity(fundamental.symbol)
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

        #           Market : Prevent close at open bar           #
        if order.entry_type == OrderType.MARKET:
            time.sleep(60 - datetime.now().second)

        while 1:  # TP limit close loop

            try:
                partial_limit(fundamental.symbol, tp_list, close_side, quantity_precision, partial_qty_divider)

            except Exception as e:
                print('Error in partial_limit :', e)
                time.sleep(1)
                continue

            TP_switch, SL_switch = False, False
            while 1:

                time.sleep(.5)

                #           Wait for bar closing time         #
                if datetime.now().second >= fundamental.bar_close_second:

                    #       Check Condition      #
                    try:
                        first_df, _ = concat_candlestick(fundamental.symbol, '1m', days=1)
                        second_df, _ = concat_candlestick(fundamental.symbol, '3m', days=1)
                        third_df, _ = concat_candlestick(fundamental.symbol, '30m', days=1)

                        df = c_profitage(first_df, second_df, third_df, label_type=LabelType.TEST)

                    except Exception as e:
                        print('Error in making Test set :', e)
                        continue

                    #           Close Condition          #
                    try:
                        #               Long                #
                        if open_side == OrderSide.BUY:
                            #           SL by Trix          #
                            if df['trix'].iloc[last_i - 1] > 0 >= df['trix'].iloc[last_i]:
                                SL_switch = True
                                print('trix info :', df['trix'].iloc[last_i - 1], df['trix'].iloc[last_i])

                        #               Short               #
                        else:
                            #           By Trix         #
                            if df['trix'].iloc[last_i - 1] < 0 <= df['trix'].iloc[last_i]:
                                SL_switch = True
                                print('trix info :', df['trix'].iloc[last_i - 1], df['trix'].iloc[last_i])

                    except Exception as e:
                        print('Error in Close Condition :', e)
                        continue

                else:
                    #          Just Take Realtime price and watch SL condition      #
                    try:
                        realtime_price = get_market_price(fundamental.symbol)
                    except Exception as e:
                        print('Error in get_market_price :', e)
                        continue

                    #           Close Condition          #
                    try:
                        #               Long                #
                        if open_side == OrderSide.BUY:

                            #            SL by Price           #
                            if realtime_price < sl_level:
                                print('realtime_price :', realtime_price)
                                SL_switch = True

                        else:
                            #           SL by Price           #
                            if realtime_price > sl_level:
                                print('realtime_price :', realtime_price)
                                SL_switch = True

                    except Exception as e:
                        print('Error in SL by Price :', e)
                        continue

                #               Public : Check Remaining Quantity             #
                try:
                    quantity = get_remaining_quantity(fundamental.symbol)
                except Exception as e:
                    print('Error in get_remaining_quantity :', e)
                    continue

                if quantity == 0.0:
                    TP_switch = True

                if TP_switch or SL_switch:
                    break
                # --- wait until close condition satisfied... ---

            print('We Got Close Signal.')

            if TP_switch:  # Trade Done!
                break
            else:
                cancel_TP = False
                re_close = False
                while 1:  # <--- This Loop for SL Close & Re-Close

                    #               Canceling Limit Order Once               #
                    if not cancel_TP:
                        #               Remaining TP close Order check            #
                        try:
                            remained_orderId = remaining_order_check()
                        except Exception as e:
                            print('Error in remaining_order_check :', e)
                            continue

                        if remained_orderId is not None:
                            #           If Remained position exist, Cancel it       #
                            try:
                                result = request_client.cancel_all_orders(symbol=fundamental.symbol)
                            except Exception as e:
                                print('Error in Cancel remained TP Close order :', e)
                                continue

                        cancel_TP = True

                    #          Get Remaining quantity         #
                    try:
                        quantity = get_remaining_quantity(fundamental.symbol)
                    except Exception as e:
                        print('Error in get_remaining_quantity :', e)
                        continue

                    #           Get price, volume precision       #
                    try:
                        _, quantity_precision = get_precision(fundamental.symbol)
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
                        if not re_close:

                            if close_side == OrderSide.SELL:
                                stop_price = calc_with_precision(realtime_price + 1 * 10 ** -price_precision, price_precision)
                            else:
                                stop_price = calc_with_precision(realtime_price - 1 * 10 ** -price_precision, price_precision)

                            #           Stop Limit Order            #
                            request_client.post_order(timeInForce=TimeInForce.GTC, symbol=fundamental.symbol,
                                                      side=close_side,
                                                      ordertype=order.sl_type, stopPrice=str(stop_price),
                                                      quantity=str(quantity), price=str(realtime_price), reduceOnly=True)

                        else:
                            #           Market Order            #
                            result = request_client.post_order(symbol=fundamental.symbol, side=close_side, ordertype=OrderType.MARKET,
                                                               quantity=str(quantity), reduceOnly=True)

                    except Exception as e:
                        print('Error in Close Order :', e)
                        re_close = True
                        continue
                    else:
                        print('Close order succeed.')

                    #       Enough time for quantity to consumed      #
                    if not re_close:
                        time.sleep(order.exit_execution_wait - datetime.now().second)
                    else:
                        time.sleep(1)

                    #               Check Remaining Quantity             #
                    try:
                        quantity = get_remaining_quantity(fundamental.symbol)
                    except Exception as e:
                        print('Error in get_remaining_quantity :', e)
                        continue

                    if quantity == 0.0:
                        print('Close order executed.')
                        break

                    else:
                        #           Re-Close            #
                        print('Re-Close')
                        re_close = True
                        continue

                break   # <--- Break for All Close Loop, Break Partial Limit Loop

        end_timestamp = int(time.time() * 1000)
        #           Get Total Income from this Trade       #
        try:
            accumulated_income += total_income(fundamental.symbol, start_timestamp, end_timestamp)
        except Exception as e:
            print('Error in total_income :', e)
        print('accumulated_income :', accumulated_income, 'USDT')
        print('accumulated_profit : %.2f %%' % (accumulated_income / start_asset * 100))
        print()
