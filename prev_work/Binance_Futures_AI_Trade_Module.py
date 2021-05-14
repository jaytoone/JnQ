from binance_futures_modules import *
from Funcs_For_Trade import *
# from Funcs_Indicator import *
# from keras_self_attention import SeqSelfAttention
from Funcs_for_TP_Ratio_TVT_Bias_Binarize_modified import profitage
from binance_futures_concat_candlestick import concat_candlestick
from Make_X_TP_Ratio_Realtime_for_Bot import made_x
from easydict import EasyDict
import gc

import tensorflow as tf
from keras.models import load_model

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))
# tf.compat.v1.disable_eager_execution()


#            Initialize Part by Symbol           #
#            First Configuration       #
with open('binance_futures_bot_config.json', 'r') as cfg:
    config = EasyDict(json.load(cfg))
fundamental = config.FUNDMTL
ai = config.AI
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

#           Make Time Interval between (Data + Train) & Bot     #
# data_train__time = (5 + 3) * 60
# time.sleep(data_train__time)

model_init = True
model_renew = False
while 1:

    #       Configuration       #
    with open('binance_futures_bot_config.json', 'r') as cfg:
        config = EasyDict(json.load(cfg))

    fundamental = config.FUNDMTL
    order = config.ORDER
    ai = config.AI

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
        with open('binance_futures_bot_config.json', 'r') as cfg:
            config = EasyDict(json.load(cfg))
        ai = config.AI

        #              thr_precision 갱신 or Not None 에만 model renew            #
        if model_init:
            model_init = False
            model_renew = True
            prev_long_thr_precision = ai.long_thr_precision
            prev_short_thr_precision = ai.short_thr_precision
        else:
            if prev_long_thr_precision != ai.long_thr_precision or \
                    prev_short_thr_precision != ai.short_thr_precision:
                prev_long_thr_precision = ai.long_thr_precision
                prev_short_thr_precision = ai.short_thr_precision
                model_renew = True
            else:
                model_renew = False

        #           Ben None thr_precision        #
        if ai.long_thr_precision is None and ai.short_thr_precision is None:
            if model_renew:
                print('ai.long_thr_precision :', ai.long_thr_precision, end=' ')
                print('ai.short_thr_precision :', ai.short_thr_precision)
            continue

        if model_renew:

            model_renew = False

            #               Model Init              #
            try:
                model
            except Exception as e:
                pass
            else:
                del model
                gc.collect()

            #               Load Model              #
            try:
                model_path = "model/rapid_ascending %s_%s_%s_futures_rnn.hdf5" % (ai.model_num, 0, 0)
                model = load_model(model_path)

            except Exception as e:
                print('Error in load_model :', e)
                quit()
            else:
                print('model loaded')

        #           Wait for bar closing time         #
        if datetime.now().second >= fundamental.bar_close_second:
            pass
        else:
            continue

        #       Prepare Test DataSet for AI Prediction      #
        try:
            first_df, _ = concat_candlestick(fundamental.symbol, '1m', ai.test_days)
            second_df, _ = concat_candlestick(fundamental.symbol, '3m', ai.test_days)
            third_df, _ = concat_candlestick(fundamental.symbol, '30m', ai.test_days)

            df = profitage(first_df, second_df, third_df, label_type=LabelType.TEST)
            print('df.index[-1] :', df.index[-1])

            Made_X, _, _ = made_x(df, ai.input_data_length, ai.scale_window_size, ai.test_data_amount)

        except Exception as e:
            print('Error in making Test set :', e)
            continue

        i = -1

        y_pred_ = model.predict(Made_X[:, :, :4], verbose=1)
        long_score = y_pred_[:, [1]]
        short_score = y_pred_[:, [2]]
        # print('long_score, short_score :', long_score, short_score)

        #           Open Condition          #
        #               Long & Short               #
        if df['box_low'].iloc[i] + abs(df['box_high'].iloc[i] - df['box_low'].iloc[i]) * \
                ai.sl_least_gap_ratio <= df['ep'].iloc[i] <= df['box_high'].iloc[i] - \
                abs(df['box_high'].iloc[i] - df['box_low'].iloc[i]) * ai.sl_least_gap_ratio:

            if ai.long_thr_precision is not None:
                if long_score[i] >= ai.long_thr_precision:
                    open_side = OrderSide.BUY

            elif ai.short_thr_precision is not None:
                if short_score[i] >= ai.short_thr_precision:
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
        ep = df['ep'].iloc[i]

    #           TP & SL             #
    partial_num = 5
    tp_ratio = 3
    partial_qty_divider = 1.5

    tp_list = list()
    if open_side == OrderSide.BUY:
        sl_level = df['long_sl'].iloc[i]

        for part_i in range(partial_num, 0, -1):
            tmp_tp_level = df['ep'].iloc[i] + abs(df['ep'].iloc[i] - df['long_sl'].iloc[i]) * tp_ratio * (
                    part_i / partial_num)
            tp_list.append(tmp_tp_level)

        if order.entry_type == OrderType.MARKET:
            if ep <= sl_level:
                print('ep <= level : %s <= %s' % (ep, sl_level))
                continue
    else:
        sl_level = df['short_sl'].iloc[i]

        for part_i in range(partial_num, 0, -1):
            tmp_tp_level = df['ep'].iloc[i] - abs(df['ep'].iloc[i] - df['short_sl'].iloc[i]) * tp_ratio * (
                    part_i / partial_num)
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
    leverage = int(target_percentage / (abs(ep - sl_level) / ep))
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
            if time.time() - exec_start > order.entry_execution_wait:
                break

            try:
                exec_quantity = get_remaining_quantity(fundamental.symbol)
            except Exception as e:
                print('Error in exec_quantity check :', e)

            if exec_quantity > quantity * order.breakout_qty_ratio:
                break

            #           realtime price compare with sl_level        #
            try:
                realtime_price = get_market_price(fundamental.symbol)
            except Exception as e:
                print('Error in get_market_price :', e)

            if open_side == OrderSide.BUY:
                if realtime_price < sl_level:
                    break
            else:
                if realtime_price > sl_level:
                    break

            time.sleep(1)

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

        close_count = 1

        #           Market : Prevent close at open bar           #
        if order.entry_type == OrderType.MARKET:
            time.sleep(70)
        while 1:  # TP limit close loop

            partial_limit(fundamental.symbol, tp_list, close_side, quantity_precision, partial_qty_divider)

            TP_switch, SL_switch = False, False
            while 1:

                #          We don't need to wait closing time,       #
                #          but we need interval to avoid API Limit      #
                time.sleep(1)

                #          Just Take Realtime price and watch SL condition      #
                try:
                    realtime_price = get_market_price(fundamental.symbol)
                except Exception as e:
                    print('Error in get_market_price :', e)
                    continue

                #           Close Condition          #
                #               Long                #
                if open_side == OrderSide.BUY:
                    #            SL by Price           #
                    if realtime_price < sl_level:
                        print('realtime_price :', realtime_price)
                        SL_switch = True

                #               Short               #
                else:
                    #           SL by Price           #
                    if realtime_price > sl_level:
                        print('realtime_price :', realtime_price)
                        SL_switch = True

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
                close_count = 0

            print('close count(remain) :', close_count)

            cancel_TP = False
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
                #           1. tp_count == 1, quantity = 1 / 2 * quantity        #
                #           2. tp_count == 0, close all remaining quantity           #
                try:
                    quantity = get_remaining_quantity(fundamental.symbol)
                except Exception as e:
                    print('Error in get_remaining_quantity :', e)
                    continue

                # if tp_count == 1:
                #     quantity *= 1 / 2

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
                    #           Stop Limit Order            #
                    request_client.post_order(timeInForce=TimeInForce.GTC, symbol=fundamental.symbol,
                                              side=close_side,
                                              ordertype=order.sl_type, stopPrice=str(sl_level),
                                              quantity=str(quantity), price=str(sl_level), reduceOnly=True)

                    #           Market Order            #
                    # result = request_client.post_order(symbol="DOTUSDT", side=OrderSide.SELL, ordertype=OrderType.MARKET,
                    #                                    quantity=str(quantity), reduceOnly=True)

                except Exception as e:
                    print('Error in Close Order :', e)
                    quit()
                    continue
                else:
                    print('Close order succeed.')

                #       Enough time for quantity to consumed      #
                time.sleep(ai.exit_execution_wait)

                #               Check Remaining Quantity             #
                try:
                    quantity = get_remaining_quantity(fundamental.symbol)
                except Exception as e:
                    print('Error in get_remaining_quantity :', e)
                    continue

                if quantity == 0.0:
                    print('Close order executed.')
                    break

                # else:
                #     #           Re-Close            #
                #     print('Re-Close')
                #     continue

            if close_count == 0:
                break  # <--- Break for All Close Loop

        end_timestamp = int(time.time() * 1000)
        #           Get Total Income from this Trade       #
        try:
            accumulated_income += total_income(fundamental.symbol, start_timestamp, end_timestamp)
        except Exception as e:
            print('Error in total_income :', e)
        print('accumulated_income :', accumulated_income, 'USDT')
        print('accumulated_profit : %.2f %%' % (accumulated_income / start_asset * 100))
        print()
