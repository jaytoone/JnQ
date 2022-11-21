from funcs_binance.binance_futures_modules import *
import time
from datetime import datetime
import numpy as np
from ast import literal_eval


def limit_order(self, order_type, limit_side, pos_side, limit_price, limit_quantity, order_info=None):
    open_retry_cnt = 0
    res_code = 0
    while 1:  # <-- loop for complete open order
        try:
            # ------------ post_order - limit & market ------------ #
            if order_type == OrderType.MARKET:
                # ------ market order ------ #
                post_order_res = request_client.post_order(symbol=self.ticker,
                                                           side=limit_side,
                                                           positionSide=pos_side,
                                                           ordertype=OrderType.MARKET,
                                                           quantity=str(limit_quantity))
            else:
                # ------ limit order - needs execution waiting time & check remaining order ------ #
                post_order_res = request_client.post_order(timeInForce=TimeInForce.GTC, symbol=self.ticker,
                                                           side=limit_side,
                                                           positionSide=pos_side,
                                                           ordertype=OrderType.LIMIT,
                                                           quantity=str(limit_quantity), price=str(limit_price))
        except Exception as e:
            status_msg = "{}".format(e)
            self.sys_log.error(status_msg)
            self.label_status.setText(status_msg)

            # ------ price precision validation ------ #
            if '-4014' in str(e):
                try:
                    realtime_price = get_market_price_v2(self.sub_client)
                    price_precision = get_precision_by_price(realtime_price)
                    limit_price = calc_with_precision(limit_price, price_precision)
                    self.sys_log.info('modified price & precision : {}, {}'.format(limit_price, price_precision))

                except Exception as e:
                    self.sys_log.error('error in get_market_price_v2 (open_order phase): {}'.format(e))
                continue

            else:
                # ------ limit_order() 에서 해결할 수 없는 error 일 경우, return ------ #
                # ------ -4003 : quantity less than zero ------ #
                if "zero" in str(e):
                    res_code = -4003

                # ------ -1111 : Precision is over the maximum defined for this asset ------ #
                #         = quantity precision error        #
                if '-1111' in str(e):
                    res_code = -1111
                    self.sys_log.info("limit_price : {}".format(limit_price))
                    self.sys_log.info("limit_quantity : {}\n".format(limit_quantity))

                # ------ -2019 : Margin is insufficient ------ #
                if '-2019' in str(e):    # ep 가 market 체결가면 발생하기도 함.
                    res_code = -2019

                return None, self.over_balance, res_code

        else:
            return post_order_res, self.over_balance, res_code


def partial_limit_order_v4(self, p_tps, p_qtys, quantity_precision):
    p_tp_idx = 0
    len_p = len(p_tps)
    post_order_res_list = []
    while 1:
        if p_tp_idx >= len_p:
            break

        p_tp = p_tps[p_tp_idx]
        if p_tp_idx == len_p - 1:
            p_qty = calc_with_precision(self.open_executedQty - np.sum(p_qtys[:-1]), quantity_precision)
        else:
            p_qty = p_qtys[p_tp_idx]

        # ------ 1. tp_order ------ #
        # try:
        #   Todo - 들고 있는 position != order_side 일 경우,  -2022,"msg":"ReduceOnly Order is rejected." 발생함
        #    1. reduceOnly=False 줘도 무관한가 => 를 줘야 무방해짐 (solved)
        post_order_res, _, res_code = limit_order(self, OrderType.LIMIT, self.close_side, self.pos_side, p_tp, p_qty)
        post_order_res_list.append(post_order_res)

        # ------ 2. check success ------ #
        if not res_code:  # if succeed, res_code = 0
            p_tp_idx += 1

        # ------ 3. errors ------ #
        else:
            status_msg = "{}".format(e)
            self.sys_log.error(status_msg)
            self.label_status.setText(status_msg)

            # ------ -1111 : quantity precision error ------ #
            if res_code == -1111:  # Todo, all_cancel 하고 처음부터 다시 (temp-solved)
                p_tp_idx = 0
                cancel_order_list(self.ticker, post_order_res_list)
                _, quantity_precision = get_precision(self.ticker)
                p_qtys = list(map(lambda x: calc_with_precision(x, quantity_precision), p_qtys))
                self.sys_log.info('p_qtys : {}'.format(p_qtys))
                continue

            # ------ -4003 : quantity less than zero ------ # Todo -> openQty 가 잘못되면 그러지 않을까, outer_scope 에서 진행해야할 것

            # except Exception as e:
            # self.sys_log.error('error in partial tp : {}'.format(e))

            # -2019 (Margin is insufficient) or something else
            # continue 하면, limit_order 내부에서 재정의할 것
            # retry_cnt += 1
            # if retry_cnt >= 10:
            #     return "maximum_retry"
            return post_order_res_list

    return post_order_res_list

def partial_limit_order_v3(self, ep, tp, close_side, pos_side, open_executedQty, price_precision, quantity_precision):

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
    self.sys_log.info('p_tps : {}'.format(p_tps))
    self.sys_log.info('p_qtys : {}'.format(p_qtys))
    
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
        post_order_res, _, res_code = limit_order(self, OrderType.LIMIT, close_side, pos_side, p_tp, p_qty)
        post_order_res_list.append(post_order_res)

        # ------ 2. check success ------ #
        if not res_code:  # if succeed, res_code = 0
            p_tp_idx += 1

        # ------ 3. errors ------ #
        # ------ -1111 : quantity precision error ------ #
        elif res_code == -1111:  # Todo, all_cancel 하고 처음부터 다시 (temp-solved)
            p_tp_idx = 0
            cancel_order_list(self.ticker, post_order_res_list)
            _, quantity_precision = get_precision(self.ticker)
            p_qtys = list(map(lambda x: calc_with_precision(x, quantity_precision), p_qtys))
            self.sys_log.info('p_qtys : {}'.format(p_qtys))
            continue

        # ------ -4003 : quantity less than zero ------ # Todo -> openQty 가 잘못되면 그러지 않을까, outer_scope 에서 진행해야할 것
        elif res_code == -4003:
            continue

        # except Exception as e:
        # self.sys_log.error('error in partial tp : {}'.format(e))

        else:  # -2019 (Margin is insufficient) or something else
            # continue 하면, limit_order 내부에서 재정의할 것
            # retry_cnt += 1
            # if retry_cnt >= 10:
            #     return "maximum_retry"
            continue

    return post_order_res_list, p_tps, p_qtys


def cancel_order(symbol, orderId):
    while 1:
        try:
            cancel_order_res = request_client.cancel_order(symbol=symbol, orderId=orderId)
        except Exception as e:
            if '-2011' in str(e):  # -2011 : "Unknown order sent." --> already canceled or filled
                return
            else:  # = 비정상 error
                self.sys_log.error("error in cancel_order : {}".format(e))
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
#                             self.sys_log.error('error in cancel_order : {}'.format(e))
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
#                     self.sys_log.error('error in get_order_info (sum(tp_executedQty) phase) : {}'.format(e))
#                 else:
#                     break
#
#     return executedPrice_list, np.sum(executedQty_list)

def market_close_order_v2(self):
    _, tp_executedQty = cancel_order_list(self.ticker, self.post_order_res_list)

    # ------------ get close_remain_quantity ------------ #
    close_remain_quantity = self.open_executedQty - tp_executedQty
    # ------------ get price, volume precision ------------ #
    #           --> reatlime 가격 변동으로 인한 precision 변동 가능성 고려      #
    _, quantity_precision = get_precision(self.ticker)
    self.sys_log.info('quantity_precision : {}'.format(quantity_precision))

    close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision)
    self.sys_log.info('close_remain_quantity : {}'.format(close_remain_quantity))

    if close_remain_quantity < 1 / (10 ** quantity_precision):
        self.label_status.setText('market rejected \n close_remain_quantity : {}'.format(close_remain_quantity))
        return

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
            post_order_res = request_client.post_order(symbol=self.ticker,
                                                       side=self.close_side,
                                                       positionSide=self.pos_side,
                                                       ordertype=OrderType.MARKET,
                                                       quantity=str(close_remain_quantity))
        except Exception as e:

            status_msg = "{}".format(e)
            self.sys_log.error(status_msg)
            self.label_status.setText(status_msg)

            # ------------ check error codes ------------ #
            # ------ -1111 : Precision is over the maximum defined for this asset ------ #
            #        = quantity precision error        #
            if '-1111' in str(e):
                code_1111 += 1
                try:
                    _, quantity_precision = get_precision(self.ticker)
                    if code_1111 % 3 == 0:
                        quantity_precision -= 1
                    elif code_1111 % 3 == 2:
                        quantity_precision += 1

                    # close_remain_quantity = get_remaining_quantity(self.ticker)
                    #   error 발생한 경우 정상 체결된 qty 는 존재하지 않는다고 가정 - close_remain_quantity 에는 변함이 없을거라 봄
                    close_remain_quantity = calc_with_precision(close_remain_quantity, quantity_precision,
                                                                def_type='floor')
                    self.sys_log.info('modified qty & precision : {} {}'.format(close_remain_quantity, quantity_precision))

                except Exception as e:
                    status_msg = "error in get modified qty_precision : {}".format(e)
                    self.sys_log.error(status_msg)
                    self.label_status.setText(status_msg)
                continue

            else:
                # ------ -2022 ReduceOnly Order is rejected ------ #
                if '-2022' in str(e):
                    error_code = '-2022'
                # ------ -4003 quantity less than zero ------ #
                if '-4003' in str(e):
                    error_code = '-4003'

                break

        else:
            self.sys_log.info('market close order enlisted')
            break

    # ------------ enough time for close_remain_quantity to be consumed ------------ #
    # if self.config.out_set.out_type != OrderType.MARKET:
    #     # ------ wait for bar ends ------ #
    #     time.sleep(self.config.trader_set.exit_execution_wait - datetime.now().second)
    #     self.sys_log.info("self.config.trader_set.exit_execution_wait - datetime.now().second : {}"
    #                  .format(self.config.trader_set.exit_execution_wait - datetime.now().second))
    #     self.sys_log.info("datetime.now().second : {}".format(datetime.now().second))
    # else:
    # time.sleep(1)  # time for qty consumed

    if error_code in ['-2022', '-4003']:    # post_order_res 가 재정의되지 않은 경우
        self.close_exec_ratio = 1.0
        self.sys_log.info('market close order executed')
        return
    else:
        #   Todo - price_list 제대로 출력하려면, order_type 맞게 명시해야함 (limit 사용 안할거니까, solved)
        market_executedPrice_list, market_executedQty = cancel_order_list(self.ticker, [post_order_res], order_type=OrderType.MARKET)
        if post_order_res.origQty - market_executedQty < 1 / (10 ** quantity_precision):
            self.close_exec_ratio = market_executedQty / post_order_res.origQty
            status_msg = "market close order executed. \n market_executedPrice_list : {}".format(market_executedPrice_list)
            self.sys_log.info(status_msg)
            # self.label_status.setText(status_msg)
            return
