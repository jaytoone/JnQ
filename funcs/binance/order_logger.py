from funcs.binance.futures_modules import *
import time
from datetime import datetime
import logging

sys_log2 = logging.getLogger()


def limit_order(self, order_type, config, limit_side, limit_price, limit_quantity,
                order_info=None, reduceonly=False):
    open_retry_cnt = 0
    res_code = 0
    while 1:  # <-- loop for complete open order
        try:
            if order_type == OrderType.MARKET:
                #           market order            #
                res_obj = request_client.post_order(symbol=config.trader_set.symbol, side=limit_side, ordertype=OrderType.MARKET,
                                          quantity=str(limit_quantity))
            else:
                #       limit order      #
                #       limit order needs execution waiting time & check remaining order         #
                res_obj = request_client.post_order(timeInForce=TimeInForce.GTC, symbol=config.trader_set.symbol,
                                          side=limit_side,
                                          ordertype=OrderType.LIMIT,
                                          quantity=str(limit_quantity), price=str(limit_price),
                                          reduceOnly=reduceonly)
        except Exception as e:
            sys_log2.error('error in limit open order : {}'.format(e))

            #        1. price precision validation      #
            if '-4014' in str(e):
                try:
                    #   Todo
                    realtime_price = get_market_price_v2(self.sub_client)
                    price_precision = get_precision_by_price(realtime_price)
                    limit_price = calc_with_precision(limit_price, price_precision)
                    sys_log2.info('modified price & precision : {}, {}'.format(limit_price, price_precision))

                except Exception as e:
                    sys_log2.error('error in get_market_price_v2 (open_order phase): {}'.format(e))
                continue

            #        -4003 : quantity less than zero        #
            if "zero" in str(e):
                res_code = -4003
                return self.over_balance, res_code

            #        -1111 : Precision is over the maximum defined for this asset   #
            #         = quantity precision error        #
            if '-1111' in str(e):
                res_code = -1111
                sys_log2.info("limit_price : {}".format(limit_price))
                sys_log2.info("limit_quantity : {}\n".format(limit_quantity))
                # break
                return self.over_balance, res_code

            #        -2019 : Margin is insufficient     #
            open_retry_cnt += 1
            if '-2019' in str(e):
                try:
                    max_available_balance = get_availableBalance()

                    #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                    available_balance, leverage = order_info

                    self.over_balance = available_balance
                    available_balance = max_available_balance * 0.9
                    #          Get available quantity         #
                    quantity = available_balance / limit_price * leverage
                    quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')
                    sys_log2.info('available_balance (temp) : {}'.format(available_balance))
                    sys_log2.info('quantity : {}'.format(quantity))

                except Exception as e:
                    sys_log2.error('error in get_availableBalance (-2019 phase) : {}'.format(e))

            elif open_retry_cnt > 100:
                sys_log2.error('open_retry_cnt over 100')
                res_code = -2019
                return self.over_balance, res_code
                # quit()
            continue

        else:
            sys_log2.info('open order enlisted : {}'.format(datetime.now()))
            # break
            return self.over_balance, res_code


def partial_limit_v2(self, config, tp_list_, close_side, quantity_precision, partial_qty_divider):
    tp_count = 0
    # retry_cnt = 0
    while 1:  # loop for partial tp
        #          get remaining quantity         #
        if tp_count == 0:
            try:
                remain_qty = get_remaining_quantity(config.trader_set.symbol)
            except Exception as e:
                sys_log2.error('error in get_remaining_quantity : {}'.format(e))
                time.sleep(config.trader_set.api_retry_term)
                continue

        #           partial tp_level        #
        tp_level = tp_list_[tp_count]

        if tp_count == len(tp_list_) - 1:
            quantity = calc_with_precision(remain_qty, quantity_precision)
            # quantity = calc_with_precision(remain_qty, quantity_precision, def_type='ceil')

        else:
            quantity = remain_qty / partial_qty_divider
            quantity = calc_with_precision(quantity, quantity_precision)

        #       1. 남을 qty 가 최소 주분 qty 보다 작고
        #       2. 반올림 주문 가능한 양이라면, remain_qty 로 order 진행    #
        if 9 / (10 ** (quantity_precision + 1)) < remain_qty - quantity < 1 / (10 ** quantity_precision):
            sys_log2.info('remain_qty, quantity (in qty < 1 / (10 ** quantity_precision) phase) : {}, {}'.format(remain_qty, quantity))

            #       Todo        #
            #        1. calc_with_precision 은 내림 상태라 r_qty 를 온전히 반영하지 못함      #
            #        2. r_qty - qty < 1 / (10 ** quantity_precision) --> 따라서, ceil 로 반영함
            # quantity = calc_with_precision(remain_qty, quantity_precision)
            quantity = calc_with_precision(remain_qty, quantity_precision, def_type='ceil')

        sys_log2.info('remain_qty, quantity : {}, {}'.format(remain_qty, quantity))

        #   Todo    #
        #    1. 남은 qty 가 최소 주문 qty_precision 보다 작다면, tp order 하지말고 return       #
        if quantity < 1 / (10 ** quantity_precision):
            return

        #           partial tp             #
        # try:
        _, res_code = limit_order(self, config.tp_set.tp_type, config, close_side, tp_level, quantity, reduceonly=True)

        if not res_code:  # non_error, res_code = 0
            tp_count += 1
            if tp_count >= len(tp_list_):
                break
            remain_qty -= quantity
            if remain_qty < 1 / (10 ** quantity_precision):  # --> means remain_qty = 0.0
                break

        elif res_code == -1111:  # continue 하면, limit_order 내부에서 재정의할 것
            continue

        #        1. -4003 : tp_list_[0] & remain_qty 로 close_order 진행       #
        #        2. 기존 주문에 추가로 주문이 가능하지 cancel order 진행하지 않고 일단, 진행         #
        elif res_code == -4003:
            tp_count = 0
            tp_list_ = [tp_list_[0]]
            continue

        # except Exception as e:
        # sys_log2.error('error in partial tp : {}'.format(e))

        else:   # -2019 (Margin is insufficient) or something else
            # continue 하면, limit_order 내부에서 재정의할 것
            # retry_cnt += 1
            # if retry_cnt >= 10:
            #     return "maximum_retry"
            continue

    return


def market_close_order(self, remain_tp_canceled, config, close_side, out, log_tp):
    while 1:  # <--- This loop for out close & complete close
        #               cancel all tp order                 #
        if not remain_tp_canceled:
            #               remaining tp order check            #
            try:
                remained_orderId = remaining_order_check(config.trader_set.symbol)

            except Exception as e:
                sys_log2.error('error in remaining_order_check : {}'.format(e))
                continue

            if remained_orderId is not None:
                #           if remained tp order exist, cancel it       #
                try:
                    request_client.cancel_all_orders(symbol=config.trader_set.symbol)

                except Exception as e:
                    sys_log2.error('error in cancel remaining tp order (out phase) : {}'.format(e))
                    continue

            remain_tp_canceled = True

        #          get remaining close_remain_quantity         #
        try:
            close_remain_quantity = get_remaining_quantity(config.trader_set.symbol)

        except Exception as e:
            sys_log2.error('error in get_remaining_quantity : {}'.format(e))
            continue

        #           get price, volume precision             #
        #           -> reatlime 가격 변동으로 인한 precision 변동 가능성       #
        try:
            _, quantity_precision = get_precision(config.trader_set.symbol)

        except Exception as e:
            sys_log2.error('error in get price & volume precision : {}'.format(e))
            continue

        else:
            sys_log2.info('quantity_precision : {}'.format(quantity_precision))

        close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision)
        sys_log2.info('close_remain_quantity : {}'.format(close_remain_quantity))

        #           close remaining close_remain_quantity             #
        #           1. order side should be opposite side to the open          #
        #           2. reduceOnly = 'true'       #

        #       Todo        #
        #        -1111 code 에 대해서 다른 대안을 제시한다면, res_code var. 로 통합가능함
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
                            sys_log2.error('error in get_market_price_v2 (close_order phase): {}'.format(e))
                            continue
                    else:
                        exit_price = log_tp

                    #           limit order             #
                    request_client.post_order(timeInForce=TimeInForce.GTC, symbol=config.trader_set.symbol,
                                              side=close_side,
                                              ordertype=config.out_set.out_type,
                                              quantity=str(close_remain_quantity), price=str(exit_price),
                                              reduceOnly=True)

                else:
                    #           market order            #
                    request_client.post_order(symbol=config.trader_set.symbol, side=close_side,
                                              ordertype=OrderType.MARKET,
                                              quantity=str(close_remain_quantity),
                                              reduceOnly=True)

            except Exception as e:
                sys_log2.error('error in close order : {}'.format(e))

                #       check error codes     #
                #       Todo        #
                #        -2022 ReduceOnly Order is rejected      #
                if '-2022' in str(e):
                    code_2022 = 1
                    break

                #        -4003 quantity less than zero
                if '-4003' in str(e):
                    sys_log2.error('error code check (-4003) : {}'.format(e))
                    break

                #       Todo        #
                #        open 이야 break 하면 되지만, close 는 아직 대응법 모름
                #        -1111 : Precision is over the maximum defined for this asset   #
                #        = quantity precision error        #
                if '-1111' in str(e):
                    code_1111 += 1
                    try:
                        _, quantity_precision = get_precision(config.trader_set.symbol)
                        if code_1111 % 3 == 0:
                            quantity_precision -= 1
                        elif code_1111 % 3 == 2:
                            quantity_precision += 1

                        close_remain_quantity = get_remaining_quantity(config.trader_set.symbol)
                        close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision,
                                                                    def_type='floor')
                        sys_log2.info('modified qty & precision : {} {}'.format(close_remain_quantity, quantity_precision))

                    except Exception as e:
                        sys_log2.error('error in get modified qty_precision : {}'.format(e))

                    continue

                config.out_set.out_type = OrderType.MARKET  # just 재시도
                continue

            else:
                sys_log2.info('market close order enlisted')
                break

        #       enough time for close_remain_quantity to be consumed      #
        if config.out_set.out_type != OrderType.MARKET:

            #       wait for bar ends      #
            time.sleep(config.trader_set.exit_execution_wait - datetime.now().second)
            sys_log2.info("config.trader_set.exit_execution_wait - datetime.now().second : {}"
                         .format(config.trader_set.exit_execution_wait - datetime.now().second))
            sys_log2.info("datetime.now().second : {}".format(datetime.now().second))

        else:
            time.sleep(1)  # time for qty consumed

        #       check remaining close_remain_quantity - after market order      #
        try:
            close_remain_quantity = get_remaining_quantity(config.trader_set.symbol)
        except Exception as e:
            sys_log2.error('error in get_remaining_quantity : {}'.format(e))
            continue

        if close_remain_quantity == 0.0 or code_2022:
            sys_log2.info('market close order executed')
            # break
            return
        else:
            #           complete close by market            #
            sys_log2.info('out_type changed to market')
            config.out_set.out_type = OrderType.MARKET
            continue
