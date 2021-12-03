from binance_funcs.binance_futures_bot_config import *
import math
import pandas as pd
# import time


def get_limit_leverage(symbol_):

    results = request_client.get_leverage_bracket()
    # leverage_dict = dict()
    for result in results:
        if result.symbol == symbol_:
            return result.brackets[0].initialLeverage

    return None


#               get balance v2              #
def get_availableBalance(asset_type='USDT'):

    results = request_client.get_balance_v2()

    # print(dir(result[0]))
    for result in results:

        # --- check asset : USDT --- #
        if result.asset == asset_type:
        # print(result.availableBalance)
            available_asset = result.availableBalance
            break

    return calc_with_precision(available_asset, 2)


# result = request_client.get_balance_v2()
# PrintMix.print_data(result)


#           realtime price v1            #
def get_market_price(symbol):

    result = request_client.get_symbol_price_ticker(symbol=symbol)
    # PrintMix.print_data(result)

    return result[0].price


#           get realtime price using websocket           #
def get_market_price_v2(sub_client):

    while 1:
        res = sub_client.connections[0].price
        if res is not None:
            return res


#           income history per pair         #
def total_income(symbol, startTime=None, endTime=None):

    results = request_client.get_income_history(symbol=symbol, startTime=startTime, endTime=endTime)
    # PrintMix.print_data(result)
    total_income_ = 0.0
    for result in results:
        total_income_ += result.income
        # print(result[i].income)

    return total_income_


#           remaining order check           #
def remaining_order_check(symbol_):

    result = request_client.get_open_orders(symbol_)
    if len(result) == 0:
        return None
    else:
        # print(result[0].orderId) #type = int
        return [result[i].orderId for i in range(len(result))]


def remaining_order_info(symbol_):

    # info_col = ['activatePrice', 'avgPrice', 'clientOrderId', 'closePosition', 'cumQuote', 'executedQty',
    #   'json_parse', 'orderId', 'origQty', 'origType', 'positionSide', 'price', 'priceRate', 'reduceOnly',
    #   'side', 'status', 'stopPrice', 'symbol', 'timeInForce', 'type', 'updateTime', 'workingType']

    result = request_client.get_open_orders(symbol_)
    if len(result) == 0:
        return None
    else:
        # print(result[0].orderId) #type = int
        #       Todo        #
        #        index set 을 0 로 설정함으로서 발생할 수 있는 이슈, 추후 고려
        return result[0]


# -------- remaining position amount check -------- #
def get_remaining_quantity(symbol_):

    results = request_client.get_position_v2()
    # r_quantity_dict = dict()
    for result in results:
        if result.symbol == symbol_:
            return abs(result.positionAmt)

    # PrintMix.print_data(result)
    # print(r_quantity_dict)

    return None


# -------- remaining position info -------- #
def get_position_info(symbol_):

    # info_col = ['entryPrice', 'isAutoAddMargin', 'isolatedMargin', 'json_parse', 'leverage', 'liquidationPrice'
    #     , 'marginType', 'markPrice', 'maxNotionalValue', 'positionAmt', 'positionSide', 'symbol', 'unrealizedProfit']

    results = request_client.get_position_v2()
    # r_quantity_dict = dict()
    for result in results:
        if result.symbol == symbol_:
            return result

    # PrintMix.print_data(result)
    # print(r_quantity_dict)

    return None


#           get precision info            #
def get_precision(symbol_):

    result = request_client.get_exchange_information()
    # members = [attr for attr in dir(result) if not callable(attr) and not attr.startswith("__")]
    # print(members)
    # precision_data =
    # precision_dict = dict()
    for data in result.symbols:
        # print(result.symbols[i].symbol)
        # print(result.symbols[i].pricePrecision)
        # print(result.symbols[i].quantityPrecision)
        # print(symbols_data[i].symbol, '->', symbols_data[i].pricePrecision, symbols_data[i].quantityPrecision)
        # precision_dict[data.symbol] = [data.pricePrecision, data.quantityPrecision]
        # print(precision_dict['BTCUSDT'])
        if data.symbol == symbol_:
            return [data.pricePrecision, data.quantityPrecision]

    return None


def calc_precision(number):

    precision_ = 0
    while 1:
        if number * (10 ** precision_) == int(number * (10 ** precision_)):
            break
        else:
            precision_ += 1

    return precision_


def calc_with_precision(data, data_precision, def_type='floor'):

    if not pd.isna(data):
        if data_precision > 0:
            if def_type == 'floor':
                data = math.floor(data * (10 ** data_precision)) / (10 ** data_precision)
            elif def_type == 'round':
                data = float(round(data, data_precision))
            else:
                data = math.ceil(data * (10 ** data_precision)) / (10 ** data_precision)
        else:
            data = int(data)

    return data


def callback(data_type: 'SubscribeMessageType', event: 'any', show_detail=False):
    if data_type == SubscribeMessageType.RESPONSE:
        if show_detail:
            print("Event ID: ", event)
    elif data_type == SubscribeMessageType.PAYLOAD:
        if show_detail:
            PrintBasic.print_obj(event)
        # sub_client.unsubscribe_all()
    else:
        print("Unknown Data:")

    if show_detail:
        print()


def error(e: 'BinanceApiException'):
    print(e.error_code + e.error_message)


def get_partial_tplist(ep, tp, open_side, partial_num, price_precision_):

    #       Todo : ultimate tp 설정       #
    #              calc_precision 씌워야함       #
    tp_list_ = []
    if open_side == OrderSide.BUY:

        for part_i in range(partial_num, 0, -1):
            partial_tp = ep + abs(tp - ep) * (part_i / partial_num)
            tp_list_.append(partial_tp)

        #   Todo : 이부분 왜 있지 ?       #
        # if order.entry_type == OrderType.MARKET:
        #     if ep <= sl_level:
        #         print('ep <= level : %s <= %s' % (ep, sl_level))
        #         continue  # while loop 가 위에 형성되면서 차후 사용을 위해선 break 으로 바뀌어야할 것

    else:

        #      Todo     #
        #       이부분이 문제네, dynamic tp 라, ep 보다 tp 가 커질 수 있음     #
        for part_i in range(partial_num, 0, -1):
            partial_tp = ep - abs(tp - ep) * (part_i / partial_num)
            tp_list_.append(partial_tp)

        # if order.entry_type == OrderType.MARKET:
        #     if ep >= sl_level:
        #         print('ep >= level : %s >= %s' % (ep, sl_level))
        #         continue

    tp_list_ = list(map(lambda x: calc_with_precision(x, price_precision_), tp_list_))

    return tp_list_


def get_partial_tplist_v2(ep, tp, partial_num, price_precision_):

    #       Todo : ultimate tp 설정       #
    #              calc_precision 씌워야함       #
    tp_list_ = []
    if tp > ep:

        for part_i in range(partial_num, 0, -1):
            partial_tp = ep + abs(tp - ep) * (part_i / partial_num)
            tp_list_.append(partial_tp)

        #   Todo : 이부분 왜 있지 ?       #
        # if order.entry_type == OrderType.MARKET:
        #     if ep <= sl_level:
        #         print('ep <= level : %s <= %s' % (ep, sl_level))
        #         continue  # while loop 가 위에 형성되면서 차후 사용을 위해선 break 으로 바뀌어야할 것

    else:

        #      Todo     #
        #       이부분이 문제네, dynamic tp 라, ep 보다 tp 가 커질 수 있음     #
        for part_i in range(partial_num, 0, -1):
            partial_tp = ep - abs(tp - ep) * (part_i / partial_num)
            tp_list_.append(partial_tp)

        # if order.entry_type == OrderType.MARKET:
        #     if ep >= sl_level:
        #         print('ep >= level : %s >= %s' % (ep, sl_level))
        #         continue

    tp_list_ = list(map(lambda x: calc_with_precision(x, price_precision_), tp_list_))

    return tp_list_


def partial_limit(symbol, tp_list_, close_side, quantity_precision, partial_qty_divider):

    tp_count = 0
    tp_type = OrderType.LIMIT
    retry_cnt = 0
    while 1:  # loop for partial tp

        #          get remaining quantity         #
        if tp_count == 0:
            try:
                remain_qty = get_remaining_quantity(symbol)
            except Exception as e:
                print('error in get_remaining_quantity :', e)
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
            print('remain_qty, quantity (in qty < 1 / (10 ** quantity_precision) phase) :', remain_qty, quantity)

            #       Todo        #
            #        1. calc_with_precision 은 내림 상태라 r_qty 를 온전히 반영하지 못함      #
            #        2. r_qty - qty < 1 / (10 ** quantity_precision) --> 따라서, ceil 로 반영함
            # quantity = calc_with_precision(remain_qty, quantity_precision)
            quantity = calc_with_precision(remain_qty, quantity_precision, def_type='ceil')

        print('remain_qty, quantity :', remain_qty, quantity)

        #   Todo    #
        #    1. 남은 qty 가 최소 주문 qty_precision 보다 작다면, tp order 하지말고 return       #
        if quantity < 1 / (10 ** quantity_precision):
            return

        #           partial tp             #
        try:
            #           limit order             #
            request_client.post_order(timeInForce=TimeInForce.GTC, symbol=symbol, side=close_side,
                                      ordertype=tp_type,
                                      quantity=str(quantity), price=str(tp_level),
                                      reduceOnly=True)
        except Exception as e:
            print('error in partial tp :', e)

            # quit()

            #        1. Quantity error occurs, tp_list_[0] & remain_qty 로 close_order 진행       #
            #        2. 기존 주문에 추가로 주문이 가능하지 cancel order 진행하지 않고 일단, 진행         #
            #        3. e 를 str 로 변환해주지 않으면, argument of type 'BinanceApiException' is not iterable error 가 발생함
            if "zero" in str(e):
                tp_count = 0
                tp_list_ = [tp_list_[0]]
                continue

            retry_cnt += 1
            if retry_cnt >= 10:
                return "maximum_retry"
            continue
        else:
            tp_count += 1
            if tp_count >= len(tp_list_):
                break
            remain_qty -= quantity
            if remain_qty < 1 / (10 ** quantity_precision):  # --> means remain_qty = 0.0
                break

    return


def get_trade_history_info(symbol_):    # pnl != percentage
    result = request_client.get_account_trades(symbol=symbol_)
    # for i in range(len(result)):
    #     print(result[i].price, result[i].realizedPnl)

    return result[-1].price


if __name__ == '__main__':

    tp_list = [91.8, 91.7, 91.65]
    partial_qty_divider = 1.5
    quantity_precision = 1
    symbol = 'XRPUSDT'
    # symbol = 'ADAUSDT'

    open_side = OrderSide.BUY
    close_side = OrderSide.SELL

    # print(get_limit_leverage(symbol))
    # print(total_income(symbol))
    # print(get_precision(symbol))


    # result = request_client.get_open_orders(symbol)
    # results = request_client.get_position_v2()
    #
    # info_col = ['entryPrice', 'isAutoAddMargin', 'isolatedMargin', 'json_parse', 'leverage', 'liquidationPrice'
    #     , 'marginType', 'markPrice', 'maxNotionalValue', 'positionAmt', 'positionSide', 'symbol', 'unrealizedProfit']
    # for result in results:
    #     if result.symbol == symbol:
    #         # print(dir(result))
    #         for ic in info_col:
    #             print(getattr(result, ic))

    # print(dir(result[0]))
    # print((result[0].price))
    # print((result[0].side))
    # info_col = ['activatePrice', 'avgPrice', 'clientOrderId', 'closePosition', 'cumQuote', 'executedQty',
    #   'json_parse', 'orderId', 'origQty', 'origType', 'positionSide', 'price', 'priceRate', 'reduceOnly',
    #   'side', 'status', 'stopPrice', 'symbol', 'timeInForce', 'type', 'updateTime', 'workingType']
    # for ic in info_col:
    #     print(getattr(result[0], ic))

    print(remaining_order_check(symbol))

    # info = request_client.futures_exchange_info()
    # PrintMix.print_data(info)
    quit()

    # if info['symbols'][0]['pair'] == tradingPairs[i]:
    # print("Price Pre ", info['symbols'][0]['pricePrecision'])

    # sub_client.subscribe_candlestick_event("btcusdt", CandlestickInterval.MIN1, callback, error)
    # print(dir(sub_client.connections[0]))
    # print(sub_client.connections.__contains__())
    # ep = 2.1762
    # quantity = 1622.59212
    # quantity_precision = 0
    # quantity = calc_with_precision(quantity, quantity_precision, def_type='floor')
    # # quantity = 1622.0
    #
    # # print("quantity :", quantity)

    # request_client.post_order(timeInForce=TimeInForce.GTC, symbol=symbol,
    #                                    side=open_side,
    #                                    ordertype='LIMIT',
    #                                    quantity=str(quantity), price=str(ep),
    #