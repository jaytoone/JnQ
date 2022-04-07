from funcs_binance.binance_futures_modules import *
import time
from datetime import datetime
import numpy as np
import logging
from ast import literal_eval

sys_log = logging.getLogger()


def limit_order(self, order_type, limit_side, limit_price, limit_quantity, order_info=None, reduceOnly=False):
    open_retry_cnt = 0
    res_code = 0
    while 1:  # <-- loop for complete open order
        try:
            # ------------ post_order - limit & market ------------ #
            if order_type == OrderType.MARKET:
                # ------ market order ------ #
                post_order_res = request_client.post_order(symbol=self.config.trader_set.symbol, side=limit_side, ordertype=OrderType.MARKET,
                                                           quantity=str(limit_quantity))
            else:
                # ------ limit order - needs execution waiting time & check remaining order ------ #
                post_order_res = request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.config.trader_set.symbol,
                                                           side=limit_side,
                                                           ordertype=OrderType.LIMIT,
                                                           quantity=str(limit_quantity), price=str(limit_price),
                                                           reduceOnly=reduceOnly)
        except Exception as e:
            # ------ limit_order() 에서 해결할 수 없는 error 일 경우, return ------ #
            # ------ -4003 : quantity less than zero ------ #
            if "zero" in str(e):
                res_code = -4003
                return None, self.over_balance, res_code

            # ------ -1111 : Precision is over the maximum defined for this asset ------ #
            #         = quantity precision error        #
            if '-1111' in str(e):
                res_code = -1111
                sys_log.info("limit_price : {}".format(limit_price))
                sys_log.info("limit_quantity : {}\n".format(limit_quantity))
                # break
                return None, self.over_balance, res_code

            sys_log.error('error in limit open order : {}'.format(e))
            # ------ price precision validation ------ #
            if '-4014' in str(e):
                try:
                    realtime_price = get_market_price_v2(self.sub_client)
                    price_precision = get_precision_by_price(realtime_price)
                    limit_price = calc_with_precision(limit_price, price_precision)
                    sys_log.info('modified price & precision : {}, {}'.format(limit_price, price_precision))

                except Exception as e:
                    sys_log.error('error in get_market_price_v2 (open_order phase): {}'.format(e))
                continue

            # ------ -2019 : Margin is insufficient ------ #
            open_retry_cnt += 1     # Todo - for tp_executedQty miscalc., self.overbalance 수정되었으면, return 으로 받아야할 것
            if '-2019' in str(e):
                try:
                    max_available_balance = get_availableBalance()

                    #    예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치    #
                    available_balance, leverage_ = order_info

                    self.over_balance = available_balance
                    available_balance = max_available_balance * 0.9
                    #          get available quantity         #
                    quantity = available_balance / limit_price * leverage_
                    quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')
                    sys_log.info('available_balance (temp) : {}'.format(available_balance))
                    sys_log.info('quantity : {}'.format(quantity))

                except Exception as e:
                    sys_log.error('error in get_availableBalance (-2019 phase) : {}'.format(e))

            elif open_retry_cnt > 100:
                sys_log.error('open_retry_cnt over 100')
                res_code = -2019
                return None, self.over_balance, res_code
                # quit()
            continue

        else:
            sys_log.info('open order enlisted : {}'.format(datetime.now()))
            return post_order_res, self.over_balance, res_code

def partial_limit_order_v3(self, ep, tp, close_side, open_executedQty, price_precision, quantity_precision, reduceOnly=True):

    p_ranges, p_qty = literal_eval(self.config.tp_set.p_ranges), literal_eval(self.config.tp_set.p_qty)
    len_p = len(p_ranges)
    en_ps, tps, open_qtys = [np.tile(arr_, (len_p, )) for arr_ in [ep, tp, open_executedQty]]
    if close_side == OrderSide.BUY:  # short
        p_tps = en_ps - (en_ps - tps) * p_ranges
    else:   # long
        p_tps = en_ps + (tps - en_ps) * p_ranges
    p_qtys = open_qtys * p_qty

    p_tps = list(map(lambda x: calc_with_precision(x, price_precision), p_tps))
    p_qtys = list(map(lambda x: calc_with_precision(x, quantity_precision), p_qtys))
    sys_log.info('p_tps : {}'.format(p_tps))
    sys_log.info('p_qtys : {}'.format(p_qtys))
    
    # retry_cnt = 0
    p_tp_idx = 0
    post_order_res_list = []
    while 1:
        if p_tp_idx >= len_p:
            break

        p_tp = p_tps[p_tp_idx]
        if p_tp_idx == len_p - 1:
            p_qty = calc_with_precision(open_executedQty - np.sum(p_qtys[:-1]), quantity_precision)
        else:
            p_qty = p_qtys[p_tp_idx]

        # ------ 1. tp_order ------ #
        # try:
        #   Todo - 들고 있는 position != order_side 일 경우,  -2022,"msg":"ReduceOnly Order is rejected." 발생함
        #    1. reduceOnly=False 줘도 무관한가 => 를 줘야 무방해짐 (solved)
        post_order_res, _, res_code = limit_order(self, self.config.tp_set.tp_type, close_side, p_tp, p_qty, reduceOnly=reduceOnly)
        post_order_res_list.append(post_order_res)

        # ------ 2. check success ------ #
        if not res_code:  # if succeed, res_code = 0
            p_tp_idx += 1

        # ------ 3. errors ------ #
        # ------ -1111 : quantity precision error ------ #
        elif res_code == -1111:  # Todo, all_cancel 하고 처음부터 다시 (temp-solved)
            p_tp_idx = 0
            cancel_order_list(self.config.trader_set.symbol, post_order_res_list)
            _, quantity_precision = get_precision(self.config.trader_set.symbol)
            p_qtys = list(map(lambda x: calc_with_precision(x, quantity_precision), p_qtys))
            sys_log.info('p_qtys : {}'.format(p_qtys))
            continue

        # ------ -4003 : quantity less than zero ------ # Todo -> openQty 가 잘못되면 그러지 않을까, outer_scope 에서 진행해야할 것
        elif res_code == -4003:
            continue

        # except Exception as e:
        # sys_log.error('error in partial tp : {}'.format(e))

        else:  # -2019 (Margin is insufficient) or something else
            # continue 하면, limit_order 내부에서 재정의할 것
            # retry_cnt += 1
            # if retry_cnt >= 10:
            #     return "maximum_retry"
            continue

    return post_order_res_list, p_tps, p_qtys

def partial_limit_order_v2(self, config, tp_list, close_side, open_executedQty, quantity_precision, partial_qty_divider, reduceOnly=True):
    tp_count = 0
    # retry_cnt = 0
    post_order_res_list = []

    #       Todo - 추후에 one_line coding 진행       #
    #        1. 미리 계산된 qty_list 작성
    #        2. 완결성 보장된 list comprehension post_ordering
    while 1:  # loop for partial tp
        # ------------ get remaining quantity ------------ #
        if tp_count == 0:
            remain_qty = open_executedQty

        #           partial tp_level        #
        tp_level = tp_list[tp_count]  # tp_count = 0 부터 post_order 시작

        if tp_count == len(tp_list) - 1:
            quantity = calc_with_precision(remain_qty, quantity_precision)
            # quantity = calc_with_precision(remain_qty, quantity_precision, def_type='ceil')
        else:
            quantity = remain_qty / partial_qty_divider
            quantity = calc_with_precision(quantity, quantity_precision)

        #       1. 남을 qty 가 최소 주분 qty 보다 작고
        #       2. 반올림 주문 가능한 양이라면, remain_qty 로 order 진행    #
        if 9 / (10 ** (quantity_precision + 1)) < remain_qty - quantity < 1 / (10 ** quantity_precision):
            sys_log.info('remain_qty, quantity (in qty < 1 / (10 ** quantity_precision) phase) : {}, {}'.format(remain_qty, quantity))

            #       Todo        #
            #        1. calc_with_precision 은 내림 상태라 r_qty 를 온전히 반영하지 못함      #
            #        2. r_qty - qty < 1 / (10 ** quantity_precision) --> 따라서, ceil 로 반영함
            # quantity = calc_with_precision(remain_qty, quantity_precision)
            quantity = calc_with_precision(remain_qty, quantity_precision, def_type='ceil')

        sys_log.info('remain_qty, quantity : {}, {}'.format(remain_qty, quantity))

        #   Todo    #
        #    1. 남은 qty 가 최소 주문 qty_precision 보다 작다면, tp order 하지말고 return       #
        if quantity < 1 / (10 ** quantity_precision):
            return post_order_res_list  # order done

        # ------------ partial tp ------------ #
        # try:
        #   Todo - 들고 있는 position != order_side,  -2022,"msg":"ReduceOnly Order is rejected." 발생함
        #    1. reduceOnly=False 줘도 무관한가 => 를 줘야 무방해짐
        post_order_res, _, res_code = limit_order(self, config.tp_set.tp_type, config, close_side, tp_level, quantity, reduceOnly=reduceOnly)
        post_order_res_list.append(post_order_res)

        # ------------ succeeds & errors ------------ #
        if not res_code:  # non_error, res_code = 0
            tp_count += 1
            if tp_count >= len(tp_list):
                break
            remain_qty -= quantity
            if remain_qty < 1 / (10 ** quantity_precision):  # --> means remain_qty = 0.0
                break

        elif res_code == -1111:  # continue 하면, limit_order 내부에서 재정의할 것
            continue

        #        1. -4003 : tp_list[0] & remain_qty 로 close_order 진행       #
        #        2. 기존 주문에 추가로 주문이 가능하지 cancel order 진행하지 않고 일단, 진행         #
        elif res_code == -4003:
            tp_count = 0
            tp_list = [tp_list[0]]
            continue

        # except Exception as e:
        # sys_log.error('error in partial tp : {}'.format(e))

        else:  # -2019 (Margin is insufficient) or something else
            # continue 하면, limit_order 내부에서 재정의할 것
            # retry_cnt += 1
            # if retry_cnt >= 10:
            #     return "maximum_retry"
            continue

    return post_order_res_list


def cancel_order(symbol, orderId):
    while 1:
        try:
            cancel_order_res = request_client.cancel_order(symbol=symbol, orderId=orderId)
        except Exception as e:
            if '-2011' in str(e):  # -2011 : "Unknown order sent." --> already canceled or filled
                return
            else:  # = 비정상 error
                sys_log.error("error in cancel_order : {}".format(e))
                continue
        else:
            return


def cancel_order_list(symbol_, post_order_res_list, order_type=OrderType.LIMIT):
    #    post_order_res_list 를 받아서 해당 order 만 취소
    #      1. None 포함된 경우는 ? - None 처리 됨
    #      2. 이미 완료된 경우 - cancel_order 가능한가 ? nope
    # ------------ cancel limit_tp orders ------------ #
    if order_type == OrderType.LIMIT:
        [cancel_order(symbol_, post_order_res.orderId) for post_order_res in post_order_res_list if post_order_res is not None]

    order_info_list = [get_order_info(symbol_, post_order_res.orderId)
                       for post_order_res in post_order_res_list if post_order_res is not None]
    # ------ get execPrice_list ------ #
    executedPrice_list = [get_execPrice(order_info) for order_info in order_info_list]
    # ------ get np.sum(executedQty_list) ------ #
    executedQty_list = [get_execQty(order_info) for order_info in order_info_list]

    return executedPrice_list, np.sum(executedQty_list)


# def cancel_order_list(symbol_, post_order_res_list, order_type=OrderType.LIMIT, price_only=False):
#     #    post_order_res_list 를 받아서 해당 order 만 취소
#     #      1. None 포함된 경우는 ? - None 처리 됨
#     #      2. 이미 완료된 경우 - cancel_order 가능한가 ? nope
#     executedPrice_list = []
#     executedQty_list = []
#     for post_order_res in post_order_res_list:  # list 가 [] 일 경우 바로 for loop ends
#         if post_order_res is not None:
#             # ------------ cancel limit_tp orders ------------ #
#             if order_type == OrderType.LIMIT and not price_only:
#                 while 1:
#                     try:
#                         cancel_order_res = request_client.cancel_order(symbol=symbol_, orderId=post_order_res.orderId)
#                     except Exception as e:
#                         if '-2011' in str(e):  # -2011 : "Unknown order sent." --> already canceled or filled
#                             break
#                         else:   # = 비정상 error
#                             sys_log.error('error in cancel_order : {}'.format(e))
#                             continue
#                     else:
#                         break
#
#             # ------------ sum(tp_executedQty) ------------ #
#             #    1. limit_tp 한 경우 - close_remain_quantity = open_executedQty - sum(tp_executedQty)
#             #    2. 안한 경우 - close_remain_quantity = open_executedQty (post_order_res_list = [])
#             #    3. order cancel & fill 된 경우 get_order_info ?
#             #       a. => 언제까지일지 모르나 살아있음
#             while 1:
#                 try:
#                     order_info_res = get_order_info(symbol_, post_order_res.orderId)
#                     if order_type == OrderType.LIMIT:
#                         executedPrice = float(order_info_res.price)
#                     else:
#                         executedPrice = float(order_info_res.avgPrice)
#                     executedPrice_list.append(executedPrice)
#                     executedQty_list.append(float(order_info_res.executedQty))
#                 except Exception as e:
#                     sys_log.error('error in get_order_info (sum(tp_executedQty) phase) : {}'.format(e))
#                 else:
#                     break
#
#     return executedPrice_list, np.sum(executedQty_list)

def market_close_order_v2(self, post_order_res_list, close_side, open_executedQty, reduceOnly=True):
    while 1:  # <--- This loop for complete close
        _, tp_executedQty = cancel_order_list(self.config.trader_set.symbol, post_order_res_list)

        # ------------ get close_remain_quantity ------------ #
        close_remain_quantity = open_executedQty - tp_executedQty

        # ------------ get price, volume precision ------------ #
        #           --> reatlime 가격 변동으로 인한 precision 변동 가능성 고려      #
        _, quantity_precision = get_precision(self.config.trader_set.symbol)
        sys_log.info('quantity_precision : {}'.format(quantity_precision))

        close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision)
        sys_log.info('close_remain_quantity : {}'.format(close_remain_quantity))

        # ------------ close remaining close_remain_quantity ------------ #
        #           1. order side should be opposite side to the open          #
        #           2. reduceOnly = 'true'       #
        #       Todo        #
        #        -1111 code 에 대해서 다른 대안을 제시한다면, res_code var. 로 통합가능함
        code_1111 = 0
        error_code = None
        while 1:  # <-- loop for complete close order
            try:
                # ------ market order - allow market only for complete out ------ #
                post_order_res = request_client.post_order(symbol=self.config.trader_set.symbol, side=close_side,
                                                           ordertype=OrderType.MARKET,
                                                           quantity=str(close_remain_quantity),
                                                           reduceOnly=reduceOnly)
            except Exception as e:
                # ------------ check error codes ------------ #
                # ------ -2022 ReduceOnly Order is rejected ------ #
                if '-2022' in str(e):
                    error_code = '-2022'
                    break

                # ------ -4003 quantity less than zero ------ #
                if '-4003' in str(e):
                    error_code = '-4003'
                    break

                sys_log.error('error in close order : {}'.format(e))    # 윗 error 는 logging 하지않음

                #   Todo, open 이야 break 하면 되지만, close 는 아직 대응법 모름
                # ------ -1111 : Precision is over the maximum defined for this asset ------ #
                #        = quantity precision error        #
                if '-1111' in str(e):
                    code_1111 += 1
                    try:
                        _, quantity_precision = get_precision(self.config.trader_set.symbol)
                        if code_1111 % 3 == 0:
                            quantity_precision -= 1
                        elif code_1111 % 3 == 2:
                            quantity_precision += 1

                        # close_remain_quantity = get_remaining_quantity(self.config.trader_set.symbol)
                        #   error 발생한 경우 정상 체결된 qty 는 존재하지 않는다고 가정 - close_remain_quantity 에는 변함이 없을거라 봄
                        close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision,
                                                                    def_type='floor')
                        sys_log.info('modified qty & precision : {} {}'.format(close_remain_quantity, quantity_precision))

                    except Exception as e:
                        sys_log.error('error in get modified qty_precision : {}'.format(e))
                    continue

                self.config.out_set.out_type = OrderType.MARKET  # just 재시도
                continue

            else:
                sys_log.info('market close order enlisted')
                break

        # ------------ enough time for close_remain_quantity to be consumed ------------ #
        if self.config.out_set.out_type != OrderType.MARKET:
            # ------ wait for bar ends ------ #
            time.sleep(self.config.trader_set.exit_execution_wait - datetime.now().second)
            sys_log.info("self.config.trader_set.exit_execution_wait - datetime.now().second : {}"
                         .format(self.config.trader_set.exit_execution_wait - datetime.now().second))
            sys_log.info("datetime.now().second : {}".format(datetime.now().second))
        else:
            time.sleep(1)  # time for qty consumed

        if error_code in ['-2022', '-4003']:    # post_order_res 가 재정의되지 않은 경우
            sys_log.info('market close order executed')
            return
        else:
            #   Todo - price_list 제대로 출력하려면, order_type 맞게 명시해야함 (limit 사용 안할거니까, solved)
            market_executedPrice_list, market_executedQty = cancel_order_list(self.config.trader_set.symbol, [post_order_res], order_type=OrderType.MARKET)
            if post_order_res.origQty - market_executedQty < 1 / (10 ** quantity_precision):
                sys_log.info('market close order executed')
                sys_log.info("market_executedPrice_list : {}".format(market_executedPrice_list))
                return market_executedPrice_list
            else:
                # ------ complete close by market ------ #
                sys_log.info('out_type changed to market')
                self.config.out_set.out_type = OrderType.MARKET
                continue

def market_close_order(self, remain_tp_canceled, post_order_res_list, close_side, open_executedQty, out, log_tp, reduceOnly=True):
    while 1:  # <--- This loop for complete close
        if not remain_tp_canceled:
            _, tp_executedQty = cancel_order_list(self.config.trader_set.symbol, post_order_res_list)
            remain_tp_canceled = True
        else:
            tp_executedQty = 0

        # ------------ get close_remain_quantity ------------ #
        close_remain_quantity = open_executedQty - tp_executedQty

        # ------------ get price, volume precision ------------ #
        #           --> reatlime 가격 변동으로 인한 precision 변동 가능성 고려      #
        _, quantity_precision = get_precision(self.config.trader_set.symbol)
        sys_log.info('quantity_precision : {}'.format(quantity_precision))

        close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision)
        sys_log.info('close_remain_quantity : {}'.format(close_remain_quantity))

        # ------------ close remaining close_remain_quantity ------------ #
        #           1. order side should be opposite side to the open          #
        #           2. reduceOnly = 'true'       #
        #       Todo        #
        #        -1111 code 에 대해서 다른 대안을 제시한다면, res_code var. 로 통합가능함
        code_1111 = 0
        error_code = None
        while 1:  # <-- loop for complete close order
            try:
                # ------ market order - allow market only for complete out ------ #
                post_order_res = request_client.post_order(symbol=self.config.trader_set.symbol, side=close_side,
                                                           ordertype=OrderType.MARKET,
                                                           quantity=str(close_remain_quantity),
                                                           reduceOnly=reduceOnly)
            except Exception as e:
                # ------------ check error codes ------------ #
                # ------ -2022 ReduceOnly Order is rejected ------ #
                if '-2022' in str(e):
                    error_code = '-2022'
                    break

                # ------ -4003 quantity less than zero ------ #
                if '-4003' in str(e):
                    error_code = '-4003'
                    break

                sys_log.error('error in close order : {}'.format(e))    # 윗 error 는 logging 하지않음

                #   Todo, open 이야 break 하면 되지만, close 는 아직 대응법 모름
                # ------ -1111 : Precision is over the maximum defined for this asset ------ #
                #        = quantity precision error        #
                if '-1111' in str(e):
                    code_1111 += 1
                    try:
                        _, quantity_precision = get_precision(self.config.trader_set.symbol)
                        if code_1111 % 3 == 0:
                            quantity_precision -= 1
                        elif code_1111 % 3 == 2:
                            quantity_precision += 1

                        # close_remain_quantity = get_remaining_quantity(self.config.trader_set.symbol)
                        #   error 발생한 경우 정상 체결된 qty 는 존재하지 않는다고 가정 - close_remain_quantity 에는 변함이 없을거라 봄
                        close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision,
                                                                    def_type='floor')
                        sys_log.info('modified qty & precision : {} {}'.format(close_remain_quantity, quantity_precision))

                    except Exception as e:
                        sys_log.error('error in get modified qty_precision : {}'.format(e))
                    continue

                self.config.out_set.out_type = OrderType.MARKET  # just 재시도
                continue

            else:
                sys_log.info('market close order enlisted')
                break

        # ------------ enough time for close_remain_quantity to be consumed ------------ #
        if self.config.out_set.out_type != OrderType.MARKET:
            # ------ wait for bar ends ------ #
            time.sleep(self.config.trader_set.exit_execution_wait - datetime.now().second)
            sys_log.info("self.config.trader_set.exit_execution_wait - datetime.now().second : {}"
                         .format(self.config.trader_set.exit_execution_wait - datetime.now().second))
            sys_log.info("datetime.now().second : {}".format(datetime.now().second))
        else:
            time.sleep(1)  # time for qty consumed

        if error_code in ['-2022', '-4003']:    # post_order_res 가 재정의되지 않은 경우
            sys_log.info('market close order executed')
            return
        else:
            #   Todo - price_list 제대로 출력하려면, order_type 맞게 명시해야함 (limit 사용 안할거니까, solved)
            market_executedPrice_list, market_executedQty = cancel_order_list(self.config.trader_set.symbol, [post_order_res], order_type=OrderType.MARKET)
            if post_order_res.origQty - market_executedQty < 1 / (10 ** quantity_precision):
                sys_log.info('market close order executed')
                sys_log.info("market_executedPrice_list : {}".format(market_executedPrice_list))
                return market_executedPrice_list
            else:
                # ------ complete close by market ------ #
                sys_log.info('out_type changed to market')
                self.config.out_set.out_type = OrderType.MARKET
                continue

