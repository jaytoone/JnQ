from binance_funcs.binance_futures_modules import *
import time
from datetime import datetime


def limit_order(self, order_type, config, limit_side, limit_price, limit_quantity, reduceonly=False):

    # print("self.over_balance (in open_order func phase) :", self.over_balance)
    # quit()
    open_retry_cnt = 0
    code_1111 = 0
    while 1:  # <-- loop for complete open order

        try:
            if order_type == OrderType.MARKET:

                #           market order            #
                request_client.post_order(symbol=config.init_set.symbol, side=limit_side, ordertype=OrderType.MARKET,
                                          quantity=str(limit_quantity))

            else:
                #       limit order      #
                #       limit order needs execution waiting time & check remaining order         #

                request_client.post_order(timeInForce=TimeInForce.GTC, symbol=config.init_set.symbol,
                                          side=limit_side,
                                          ordertype=OrderType.LIMIT,
                                          quantity=str(limit_quantity), price=str(limit_price),
                                          reduceOnly=reduceonly)
        except Exception as e:
            print('error in limit open order :', e)

            #        1. price precision validation      #
            if '-4014' in str(e):
                try:
                    #   Todo
                    realtime_price = get_market_price_v2(self.sub_client)
                    price_precision = calc_precision(realtime_price)
                    limit_price = calc_with_precision(limit_price, price_precision)
                    print('modified price & precision :', limit_price, price_precision)

                except Exception as e:
                    print('error in get_market_price_v2 (open_order phase):', e)

                continue

            #        -1111 : Precision is over the maximum defined for this asset   #
            #         = quantity precision error        #
            if '-1111' in str(e):
                code_1111 = 1
                print("limit_price :", limit_price)
                print("limit_quantity :", limit_quantity)
                print()
                # break
                return self.over_balance, code_1111

            open_retry_cnt += 1

            #        -2019 : Margin is insufficient     #
            if '-2019' in str(e):
                try:
                    max_available_balance = get_availableBalance()
                    #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                    #   Todo
                    self.over_balance = available_balance
                    available_balance = max_available_balance * 0.9
                    #          Get available quantity         #
                    quantity = available_balance / limit_price * leverage
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
            # break
            return self.over_balance, code_1111


def market_close_order(self, remain_tp_canceled, config, close_side, out, log_tp):

    while 1:  # <--- This loop for out close & complete close

        #               cancel all tp order                 #
        if not remain_tp_canceled:
            #               remaining tp order check            #
            try:
                remained_orderId = remaining_order_check(config.init_set.symbol)

            except Exception as e:
                print('error in remaining_order_check :', e)
                continue

            if remained_orderId is not None:
                #           if remained tp order exist, cancel it       #
                try:
                    request_client.cancel_all_orders(symbol=config.init_set.symbol)

                except Exception as e:
                    print('error in cancel remaining tp order (out phase) :', e)
                    continue

            remain_tp_canceled = True

        #          get remaining close_remain_quantity         #
        try:
            close_remain_quantity = get_remaining_quantity(config.init_set.symbol)

        except Exception as e:
            print('error in get_remaining_quantity :', e)
            continue

        #           get price, volume precision             #
        #           -> reatlime 가격 변동으로 인한 precision 변동 가능성       #
        try:
            _, quantity_precision = get_precision(config.init_set.symbol)

        except Exception as e:
            print('error in get price & volume precision :', e)
            continue

        else:
            print('quantity_precision :', quantity_precision)

        close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision)
        print('close_remain_quantity :', close_remain_quantity)

        #           close remaining close_remain_quantity             #
        #           1. order side should be opposite side to the open          #
        #           2. reduceOnly = 'true'       #
        code_1111 = 0
        code_2022 = 0
        while 1:  # <-- loop for complete close order
            try:
                #        1. limit out 은 한동안 없을 것
                if config.out_set.out_type != OrderType.MARKET:

                    if pd.isna(out):

                        try:
                            #   Todo
                            exit_price = get_market_price_v2(self.sub_client)
                        except Exception as e:
                            print('error in get_market_price_v2 (close_order phase):', e)
                            continue

                    else:
                        exit_price = log_tp

                    #           limit order             #
                    request_client.post_order(timeInForce=TimeInForce.GTC, symbol=config.init_set.symbol,
                                              side=close_side,
                                              ordertype=config.out_set.out_type,
                                              quantity=str(close_remain_quantity), price=str(exit_price),
                                              reduceOnly=True)

                else:
                    #           market order            #
                    request_client.post_order(symbol=config.init_set.symbol, side=close_side,
                                              ordertype=OrderType.MARKET,
                                              quantity=str(close_remain_quantity),
                                              reduceOnly=True)

            except Exception as e:
                print('error in close order :', e)

                #       check error codes     #
                #       Todo        #
                #        -2022 ReduceOnly Order is rejected      #
                if '-2022' in str(e):
                    code_2022 = 1
                    break

                #        -4003 quantity less than zero ?
                if '-4003' in str(e):
                    print('error code check (-4003) :', str(e))
                    break

                #       Todo        #
                #        open 이야 break 하면 되지만, close 는 아직 대응법 모름
                #        -1111 : Precision is over the maximum defined for this asset   #
                #        = quantity precision error        #
                if '-1111' in str(e):
                    code_1111 += 1
                    try:
                        _, quantity_precision = get_precision(config.init_set.symbol)
                        if code_1111 % 3 == 0:
                            quantity_precision -= 1
                        elif code_1111 % 3 == 2:
                            quantity_precision += 1

                        close_remain_quantity = get_remaining_quantity(config.init_set.symbol)
                        close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision,
                                                                    def_type='floor')
                        print('modified qty & precision :', close_remain_quantity, quantity_precision)

                    except Exception as e:
                        print('error in get modified qty_precision :', e)

                    continue

                config.out_set.out_type = OrderType.MARKET  # just 재시도
                continue

            else:
                print('market close order enlisted')
                break

        #       enough time for close_remain_quantity to be consumed      #
        if config.out_set.out_type != OrderType.MARKET:

            #       wait for bar ends      #
            time.sleep(config.out_set.exit_execution_wait - datetime.now().second)
            print("config.out_set.exit_execution_wait - datetime.now().second :",
                  config.out_set.exit_execution_wait - datetime.now().second)
            print("datetime.now().second :", datetime.now().second)

        else:
            time.sleep(1)  # time for qty consumed

        #       check remaining close_remain_quantity - after market order      #
        try:
            close_remain_quantity = get_remaining_quantity(config.init_set.symbol)
        except Exception as e:
            print('error in get_remaining_quantity :', e)
            continue

        if close_remain_quantity == 0.0 or code_2022:
            print('market close order executed')
            # break
            return

        else:
            #           complete close by market            #
            print('out_type changed to market')
            config.out_set.out_type = OrderType.MARKET
            continue
